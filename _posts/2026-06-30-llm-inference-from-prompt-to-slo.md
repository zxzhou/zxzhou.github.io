---
layout: post
title: "LLM Inference, From Prompt to SLO"
date: 2026-06-30
tags: [llm, inference, performance, systems]
excerpt: "Study notes from the LLM Inference Handbook: why inference is a request lifecycle problem before it is a model problem, and how TTFT, ITL, TPS, and goodput map to user experience."
---

The useful way to study LLM inference is to stop treating it as one model call.

An inference request is a pipeline. Text becomes tokens, tokens become attention states, attention states become the next token, and the serving system repeats that loop until the response stops. The [LLM Inference Handbook](https://handbook.modular.com/index.md) is useful because it keeps pulling the reader back to this full path: core model mechanics, GPU memory, batching, routing, kernel behavior, and operations all participate in the latency a user feels.

The first mental shift is simple: inference is where the model meets a service-level objective. Training asks how to produce weights. Inference asks how to make those weights answer many different users, under changing load, with bounded latency and cost.

## The request has two different phases

The handbook's page on [how LLM inference works](https://handbook.modular.com/llm-inference-basics/how-does-llm-inference-work.md) separates the request into prefill and decode.

During prefill, the model processes the whole prompt. This phase is parallelizable across the input tokens, so it tends to behave like a large matrix-compute problem. Long prompts make prefill expensive because the system has to turn the full context into internal attention state.

During decode, the model generates one token at a time. Each new token depends on the tokens before it. This phase is sequential at the request level, and it keeps reading and writing the KV cache. That is why decode often becomes memory-bandwidth sensitive even when the GPU has impressive peak FLOPS.

This distinction explains a lot of production behavior. A chatbot with short prompts and long answers has different pressure from a retrieval system that sends huge context and generates a short JSON object. They may use the same model and still need different batching, routing, and hardware choices.

Here is a tiny request model that makes the split explicit:

```python
from dataclasses import dataclass


@dataclass
class InferenceRequest:
    prompt_tokens: int
    output_tokens: int
    prefill_ms_per_token: float
    decode_ms_per_token: float
    queue_ms: float = 0.0

    @property
    def ttft_ms(self) -> float:
        return self.queue_ms + self.prompt_tokens * self.prefill_ms_per_token

    @property
    def generation_ms(self) -> float:
        return self.output_tokens * self.decode_ms_per_token

    @property
    def e2e_ms(self) -> float:
        return self.ttft_ms + self.generation_ms


req = InferenceRequest(
    prompt_tokens=4_000,
    output_tokens=600,
    prefill_ms_per_token=0.08,
    decode_ms_per_token=18.0,
    queue_ms=35.0,
)

print(f"TTFT: {req.ttft_ms:.0f} ms")
print(f"E2E latency: {req.e2e_ms / 1000:.1f} s")
```

The numbers are illustrative, but the shape is the point. TTFT is mostly queueing plus prefill. End-to-end latency includes decode, and decode grows with output length. A single average latency number hides which part of the system actually needs work.

## Metrics are a map of user pain

The handbook's [metrics page](https://handbook.modular.com/llm-inference-basics/llm-inference-metrics.md) gives the vocabulary most teams need:

| Metric | What it tells you |
|---|---|
| TTFT | How long the user waits before seeing the first token |
| ITL | The gap between generated tokens after streaming starts |
| TPOT | Time per output token, often used as a decode-speed metric |
| E2EL | Full request latency from submission to completed response |
| TPS | Aggregate token throughput across requests |
| Goodput | Completed requests per second that meet the SLO |

The important metric is goodput. Raw throughput can improve while product quality gets worse. A server can push more total tokens per second by over-batching, but if P99 TTFT crosses the user-facing SLO, the extra throughput is not useful service capacity.

This is the same lesson as ordinary backend systems, but LLMs make it easier to fool yourself. Token throughput feels like an objective number. User experience is shaped by the distribution: queueing, first token, streaming smoothness, and completion time.

One way to keep yourself honest is to calculate goodput directly from traces:

```python
from statistics import quantiles


def summarize_requests(requests, ttft_slo_ms=800, e2e_slo_ms=8_000):
    total = len(requests)
    ok = [
        r for r in requests
        if r["ttft_ms"] <= ttft_slo_ms and r["e2e_ms"] <= e2e_slo_ms
    ]
    window_seconds = max(r["completed_at"] for r in requests) - min(
        r["started_at"] for r in requests
    )
    p99_ttft = quantiles([r["ttft_ms"] for r in requests], n=100)[98]

    return {
        "request_count": total,
        "slo_pass_rate": len(ok) / total,
        "goodput_rps": len(ok) / window_seconds,
        "p99_ttft_ms": p99_ttft,
    }
```

This small function changes the tuning conversation. You stop asking only whether a new configuration increases TPS. You also ask whether the extra tokens preserve the latency contract.

## The lifecycle view prevents local optimization

Once prefill, decode, and goodput are visible, many handbook topics stop looking like separate tricks.

[Continuous batching](https://handbook.modular.com/inference-optimization/static-dynamic-continuous-batching.md) improves utilization because requests enter and leave the active batch at token boundaries. [Prefix caching](https://handbook.modular.com/inference-optimization/prefix-caching.md) improves TTFT when many requests share a system prompt or chat prefix. [PagedAttention](https://handbook.modular.com/inference-optimization/pagedattention.md) increases useful concurrency by reducing KV cache fragmentation. [Prefill-decode disaggregation](https://handbook.modular.com/inference-optimization/prefill-decode-disaggregation.md) exists because prefill and decode stress hardware differently.

These are not isolated optimizations. They are attempts to remove pressure from different stages of the same request lifecycle.

That framing also explains why choosing an inference framework before understanding the workload is premature. The relevant question is not only which framework is fastest in a benchmark. It is which framework gives control over the bottleneck your workload will actually hit: prefill, decode, KV cache memory, routing, cold starts, observability, or deployment operations.

## My study note

The first level of inference understanding is vocabulary. The second level is phase separation. Once you can look at a slow request and say "this is queueing," "this is prefill," "this is decode," or "this is a goodput problem," the handbook becomes easier to navigate.

The core model is not complicated:

```text
user request
  -> queueing
  -> tokenization
  -> prefill over prompt tokens
  -> decode one token at a time
  -> detokenization and streaming
  -> completed response
```

Everything after that is engineering pressure management. Memory decides how many requests can stay alive. Scheduling decides which tokens run next. Kernels decide how efficiently the GPU moves data and performs matrix work. Routing decides which replica receives the request. Observability decides whether any of this remains visible in production.

Inference is not one call. It is a small distributed system wrapped around a sequential token loop.
