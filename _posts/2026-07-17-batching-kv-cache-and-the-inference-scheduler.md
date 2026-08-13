---
layout: post
title: "Batching, KV Cache, and the Inference Scheduler"
date: 2026-07-17
tags: [llm, inference, scheduling, batching, kv-cache]
excerpt: "Study notes from the LLM Inference Handbook: continuous batching, PagedAttention, prefix caching, and why the inference scheduler is where GPU utilization becomes product latency."
---

The scheduler is the hidden center of an LLM serving system.

Model architecture decides what computation must happen. Hardware decides what computation is possible. The scheduler decides which request gets the next slice of GPU work. That decision shapes throughput, TTFT, streaming smoothness, memory pressure, and fairness.

The [LLM Inference Handbook](https://handbook.modular.com/index.md) covers this through [static, dynamic, and continuous batching](https://handbook.modular.com/inference-optimization/static-dynamic-continuous-batching.md), [PagedAttention](https://handbook.modular.com/inference-optimization/pagedattention.md), [prefix caching](https://handbook.modular.com/inference-optimization/prefix-caching.md), and [chunked prefill](https://handbook.modular.com/inference-optimization/static-dynamic-continuous-batching.md#chunked-prefill). Read together, these pages make one thing clear: inference performance is less about sending one request through the model quickly and more about keeping the GPU usefully busy while many requests are at different stages of life.

## Static batching wastes time on shape mismatch

Static batching groups requests into fixed batches and runs them together. This is simple and useful for offline workloads where inputs are known ahead of time.

It is awkward for online inference. Requests arrive at different times. Prompts have different lengths. Outputs stop at different lengths. If the server waits too long to fill a batch, TTFT gets worse. If it runs small batches immediately, GPU utilization suffers. If it pads everything to the longest sequence, compute is wasted on padding tokens.

Dynamic batching improves this by forming batches from requests that arrive within a short window. That helps, but the batch is still a batch: once it starts, short requests and long requests remain coupled.

Continuous batching is the more important idea for LLM serving. Requests can join and leave the active set at token boundaries. When one request finishes, another can enter. The server no longer has to wait for the longest generation in a batch before admitting new work.

The Handbook's batching page also calls out [padding tokens](https://handbook.modular.com/inference-optimization/static-dynamic-continuous-batching.md#what-are-padding-tokens-in-llm-batching) and ragged shapes. A small padding calculator shows why static batching becomes wasteful when lengths vary:

```python
def padding_waste(sequence_lengths):
    padded_len = max(sequence_lengths)
    real_tokens = sum(sequence_lengths)
    padded_tokens = padded_len * len(sequence_lengths)
    return {
        "real_tokens": real_tokens,
        "padded_tokens": padded_tokens,
        "wasted_tokens": padded_tokens - real_tokens,
        "waste_ratio": (padded_tokens - real_tokens) / padded_tokens,
    }


print(padding_waste([128, 256, 1024, 4096]))
```

This is not an LLM benchmark. It is the shape problem that batching policy has to manage. Online LLM traffic often mixes short and long prompts, and output lengths stop at different times.

## Decode turns scheduling into a token-level problem

Autoregressive decode produces one token per active request per step. That means the scheduler has repeated opportunities to change the active batch. Conceptually:

```python
from collections import deque


waiting = deque()
active = []
max_active = 64


def scheduler_step():
    while waiting and len(active) < max_active:
        active.append(waiting.popleft())

    batch = [req.current_token_ids for req in active]
    next_tokens = model_decode_step(batch)

    still_running = []
    for req, token in zip(active, next_tokens):
        req.append(token)
        if not req.is_finished():
            still_running.append(req)

    active[:] = still_running
```

Real engines are much more sophisticated, but this toy loop captures the scheduling surface. Each decode step is a chance to fill empty slots, preserve streaming cadence, and respect memory limits.

The scheduler also has to deal with prefill. A long prefill can occupy enough compute to delay decode steps for already-streaming requests. That creates visible jitter. The handbook's [chunked prefill](https://handbook.modular.com/inference-optimization/static-dynamic-continuous-batching.md#chunked-prefill) section explains the remedy: split a long prefill into chunks so active decodes can continue between chunks.

A slightly more explicit scheduler has two queues: requests waiting for prefill and requests already decoding. The policy below gives decode a regular slot, then admits a bounded amount of prefill work. This mirrors the Handbook's point that chunked prefill prevents a whole long prompt from blocking active decodes.

```python
from collections import deque


prefill_queue = deque()
decode_active = []
MAX_PREFILL_TOKENS_PER_STEP = 2048


def step_scheduler():
    if decode_active:
        run_one_decode_step(decode_active)

    budget = MAX_PREFILL_TOKENS_PER_STEP
    while prefill_queue and budget > 0:
        req = prefill_queue[0]
        chunk = min(req.remaining_prefill_tokens, budget)
        run_prefill_chunk(req, chunk)
        req.remaining_prefill_tokens -= chunk
        budget -= chunk

        if req.remaining_prefill_tokens == 0:
            prefill_queue.popleft()
            decode_active.append(req)
        else:
            break
```

The implementation details vary by runtime. The policy question is stable: how much prefill can the server admit without damaging ITL for requests that are already streaming?

## KV cache allocation is scheduler capacity

The scheduler can only keep a request active if memory exists for its KV cache.

Traditional contiguous KV allocation over-reserves memory because each request may need a different final sequence length. It is similar to reserving one huge continuous block for every process. Some blocks are partially empty. Other blocks cannot be reused because the gaps are the wrong size.

[PagedAttention](https://handbook.modular.com/inference-optimization/pagedattention.md) changes that allocation model. It stores KV cache in fixed-size blocks, so a sequence can grow by acquiring more blocks instead of requiring one contiguous reservation. This makes memory usage more granular and reduces fragmentation.

That matters to the scheduler because available KV-cache space determines active concurrency. Better allocation turns stranded memory into usable request slots.

The operating-system analogy is helpful, but the production result is more concrete: more active sequences can fit in the same VRAM, and variable-length requests become less punishing.

The block-table idea can be sketched without implementing attention:

```python
import math


class KVBlockAllocator:
    def __init__(self, total_blocks, block_tokens):
        self.free = list(range(total_blocks))
        self.block_tokens = block_tokens
        self.tables = {}

    def allocate_for_sequence(self, request_id, total_tokens):
        needed = math.ceil(total_tokens / self.block_tokens)
        if needed > len(self.free):
            raise MemoryError("not enough KV blocks")
        blocks = [self.free.pop() for _ in range(needed)]
        self.tables[request_id] = blocks
        return blocks

    def append_tokens(self, request_id, new_total_tokens):
        current_blocks = len(self.tables[request_id])
        needed = math.ceil(new_total_tokens / self.block_tokens)
        for _ in range(needed - current_blocks):
            self.tables[request_id].append(self.free.pop())

    def release(self, request_id):
        self.free.extend(self.tables.pop(request_id))
```

The real serving engine also maps these blocks into attention kernels. The concept that matters here is allocation granularity. A request can grow by adding blocks, and finished requests return blocks to the pool.

## Prefix caching turns repetition into capacity

Many LLM applications repeat large prompt prefixes:

```text
system prompt
developer instructions
tool schema
few-shot examples
retrieved policy text
user-specific stable context
new user question
```

Without prefix caching, the server recomputes the shared prefix for each request. With [prefix caching](https://handbook.modular.com/inference-optimization/prefix-caching.md), the server can reuse the KV cache for the shared prefix and only compute the new suffix.

Prompt layout becomes an inference optimization. Stable, reusable content should appear before volatile content. Two semantically identical prompts with different ordering can have different cache behavior.

A simple prompt builder can make that policy explicit:

```python
def build_prompt(system_prompt, tool_schemas, examples, user_profile, user_query):
    stable_prefix = "\n\n".join([
        system_prompt,
        render_tool_schemas(tool_schemas),
        render_examples(examples),
    ])
    semi_stable_context = render_user_profile(user_profile)

    return "\n\n".join([
        stable_prefix,
        semi_stable_context,
        f"User request:\n{user_query}",
    ])
```

The intent is to keep the highest-reuse tokens as early and as stable as possible. That increases cache-hit probability across requests and sessions.

This has an architectural implication. Prompt engineering is usually discussed as a quality technique. In production inference, prompt structure also affects TTFT, memory pressure, and cost.

The prefix-cache key is usually derived from tokens, not from the human-readable text after lossy normalization. A toy cache interface makes the failure mode visible:

```python
import hashlib


def prefix_key(token_ids):
    payload = ",".join(map(str, token_ids)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def split_stable_prefix(tokenizer, system_prompt, tools, examples):
    prefix_text = "\n\n".join([
        system_prompt,
        render_tool_schemas(tools),
        render_examples(examples),
    ])
    return tokenizer.encode(prefix_text)


stable_tokens = split_stable_prefix(tokenizer, system_prompt, tools, examples)
cache_key = prefix_key(stable_tokens)
```

The Handbook's [prefix-caching page](https://handbook.modular.com/inference-optimization/prefix-caching.md#how-to-structure-prompts-for-maximum-cache-hits) frames the practical rule: keep stable shared content in a consistent prefix. Small formatting differences can turn a reusable prefix into a cache miss.

## Speculation changes the scheduler contract

[Speculative decoding](https://handbook.modular.com/inference-optimization/speculative-decoding.md) adds another scheduling shape. A draft model proposes multiple tokens. The target model verifies them. When the target accepts several draft tokens, decode advances by more than one token of output. When acceptance is low, the server has done extra draft work with less benefit.

The scheduler-level accounting looks like this:

```python
def speculative_step(request, draft_model, target_model, max_draft_tokens):
    draft = draft_model.propose(request.tokens, max_tokens=max_draft_tokens)
    accepted = target_model.verify(request.tokens, draft)

    request.tokens.extend(accepted)
    return {
        "draft_tokens": len(draft),
        "accepted_tokens": len(accepted),
        "acceptance_rate": len(accepted) / len(draft) if draft else 0.0,
    }
```

The Handbook emphasizes acceptance rate, memory overhead, and wasted compute. That turns speculation into a workload-specific optimization rather than a universal switch.

## My study note

The scheduler sits at the intersection of three resources:

```text
compute:
  which tokens run in this step?

memory:
  which requests can keep KV cache resident?

latency:
  which users are waiting for first token or next token?
```

Static batching optimizes for simplicity. Dynamic batching reduces idle time. Continuous batching treats generation as a stream of token-level scheduling decisions. Chunked prefill prevents long prompts from blocking active decodes for too long. PagedAttention improves the memory allocator that the scheduler depends on. Prefix caching avoids repeated prefill work when prompts share a stable beginning.

The phrase I would keep in my notes is this: the scheduler converts GPU utilization into product latency.

That is why inference frameworks matter. The framework is not only a wrapper around `forward()`. It owns batching policy, memory layout, cache reuse, queueing, cancellation, streaming, and observability hooks. For production LLM systems, that is the serving product.
