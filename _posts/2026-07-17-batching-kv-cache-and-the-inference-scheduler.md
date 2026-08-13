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

## KV cache allocation is scheduler capacity

The scheduler can only keep a request active if memory exists for its KV cache.

Traditional contiguous KV allocation over-reserves memory because each request may need a different final sequence length. It is similar to reserving one huge continuous block for every process. Some blocks are partially empty. Other blocks cannot be reused because the gaps are the wrong size.

[PagedAttention](https://handbook.modular.com/inference-optimization/pagedattention.md) changes that allocation model. It stores KV cache in fixed-size blocks, so a sequence can grow by acquiring more blocks instead of requiring one contiguous reservation. This makes memory usage more granular and reduces fragmentation.

That matters to the scheduler because available KV-cache space determines active concurrency. Better allocation turns stranded memory into usable request slots.

The operating-system analogy is helpful, but the production result is more concrete: more active sequences can fit in the same VRAM, and variable-length requests become less punishing.

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
