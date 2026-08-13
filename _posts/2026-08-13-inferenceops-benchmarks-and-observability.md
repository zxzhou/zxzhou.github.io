---
layout: post
title: "InferenceOps: Benchmarks, Observability, and Framework Choice"
date: 2026-08-13
tags: [llm, inferenceops, observability, benchmarking, production-ai]
excerpt: "Study notes from the LLM Inference Handbook: production inference needs workload-specific benchmarks, visible SLOs, and runtime choices that match the bottleneck."
---

Inference work becomes real when a model has to survive production.

At that point, the interesting questions change. Can the system meet TTFT and end-to-end latency SLOs under bursty traffic? Which requests miss goodput? How much does a long context cost? Which model version increased output length? Which worker has KV-cache pressure? Which runtime falls back to a slower kernel? Which deployment can roll back safely?

The [LLM Inference Handbook](https://handbook.modular.com/index.md) covers this operating layer through [performance benchmarks](https://handbook.modular.com/inference-optimization/llm-performance-benchmarks.md), [observability](https://handbook.modular.com/infrastructure-and-operations/comprehensive-observability.md), [InferenceOps management](https://handbook.modular.com/infrastructure-and-operations/inferenceops-and-management.md), and [inference framework choice](https://handbook.modular.com/getting-started/choosing-the-right-inference-framework.md). The main lesson is that production inference is a measurement problem as much as a serving problem.

## Benchmarks need the shape of your workload

Generic benchmarks are useful for orientation. They are weak evidence for your deployment.

An LLM serving workload has a shape:

```text
prompt length distribution
output length distribution
concurrency
burst pattern
streaming or non-streaming
shared-prefix rate
model mix
tool/function-call rate
SLO threshold
region and network path
```

Change any of these and the "fastest" runtime or configuration can change. A short-output extraction workload and a long-form chat workload may stress different parts of the same engine. A prefix-heavy agent workload may reward cache locality. A retrieval workload with large context may be dominated by prefill. A high-concurrency chat service may care most about decode cadence and P99 TTFT.

This is why I would write benchmarks as small experiments with explicit workload contracts:

```yaml
name: customer_support_chat_v1
model: qwen2.5-14b-instruct
traffic:
  concurrent_users: [16, 32, 64, 128]
  arrival_pattern: poisson
tokens:
  prompt_p50: 1800
  prompt_p95: 6500
  output_p50: 220
  output_p95: 900
cache:
  shared_system_prompt: true
  avg_prefix_reuse_rate: 0.45
slo:
  ttft_p99_ms: 1200
  e2e_p99_ms: 10000
```

The benchmark result should answer a deployment question, not produce a trophy number.

## Observability must cross model, runtime, and infrastructure

The handbook's [observability page](https://handbook.modular.com/infrastructure-and-operations/comprehensive-observability.md) frames LLM observability as metrics, logs, and events across infrastructure, application, and model layers. That layering matters because a bad user experience can come from any of them.

A practical metric set should include:

| Layer | Signals |
|---|---|
| Product | success rate, error rate, user-visible latency, cancellation |
| Inference | TTFT, ITL, E2EL, TPS, goodput, queue time |
| Token shape | prompt tokens, output tokens, total tokens, stop reason |
| Scheduler | active sequences, waiting requests, batch size, prefill/decode mix |
| Memory | GPU memory, KV-cache usage, cache hit rate, fragmentation symptoms |
| Hardware | SM utilization, HBM bandwidth, interconnect traffic |
| Quality | refusal rate, schema validity, tool-call success, eval score |

The reason to collect these together is diagnosis. A rising P99 TTFT means little by itself. It becomes actionable when you can see whether prompt lengths increased, prefix-cache hit rate dropped, queue time grew, a model version changed, or one worker pool hit memory pressure.

Here is the kind of instrumentation surface I would want around a serving gateway:

```python
def record_inference_metrics(metrics, request, response, worker):
    metrics.histogram("llm.prompt_tokens").observe(request.prompt_tokens)
    metrics.histogram("llm.output_tokens").observe(response.output_tokens)
    metrics.histogram("llm.ttft_ms").observe(response.ttft_ms)
    metrics.histogram("llm.e2e_ms").observe(response.e2e_ms)
    metrics.histogram("llm.queue_ms").observe(response.queue_ms)

    metrics.counter("llm.requests_total", {
        "model": request.model,
        "route": worker.route_group,
        "status": response.status,
    }).inc()

    metrics.gauge("llm.worker.kv_cache_used_bytes", {
        "worker": worker.id,
        "model": request.model,
    }).set(worker.kv_cache_used_bytes)

    metrics.gauge("llm.worker.active_sequences", {
        "worker": worker.id,
    }).set(worker.active_sequences)
```

This is not glamorous code. It is the code that lets a team improve inference without guessing.

## Framework choice is an operational decision

The handbook's [framework-choice page](https://handbook.modular.com/getting-started/choosing-the-right-inference-framework.md) is useful because it treats inference frameworks as serving systems, not just model loaders.

Library mode and server mode have different trade-offs. Library mode gives application-level control and easier custom integration. Server mode gives a clearer serving boundary, standardized APIs, independent scaling, and operational isolation.

The right choice depends on which surfaces you need to own:

```text
Do you need custom batching policy?
Do you need OpenAI-compatible or Anthropic-compatible APIs?
Do you need tensor parallelism or expert parallelism?
Do you need prefix caching, speculative decoding, or PagedAttention?
Do you need metrics at the scheduler layer?
Do you need multi-model routing?
Do you need safe rolling updates?
Do you need to swap runtimes per model family?
```

This is close to the agent-harness argument I keep returning to in other writing: the surrounding system matters. A stronger model does not remove the need for the serving layer. Better kernels and runtimes change the boundary, but someone still owns queueing, routing, metrics, deploys, rollback, and cost.

## InferenceOps is feedback loop design

[InferenceOps](https://handbook.modular.com/infrastructure-and-operations/inferenceops-and-management.md) is the discipline around standardized deployments, safe updates, centralized management, fault tolerance, and cost control. The name sounds abstract until something goes wrong.

The real loop looks like this:

```text
deploy model/runtime/config
  -> observe workload and SLO behavior
  -> identify bottleneck
  -> tune batching, memory, routing, precision, or hardware
  -> benchmark against the workload contract
  -> roll out safely
  -> keep the trace for the next regression
```

This loop is where inference teams compound. A one-off benchmark helps one decision. A benchmark suite plus observability turns production behavior into reusable knowledge. That is how teams learn whether a new quantization format, routing policy, or framework upgrade actually improves the service.

## My study note

The production view of inference can be compressed into one rule:

```text
Optimize against SLO-qualified workload behavior, not abstract throughput.
```

That rule pushes you toward four habits. First, define the workload shape before benchmarking. Second, measure goodput and P99 latency, not only TPS. Third, instrument the scheduler and KV cache, because many failures hide below the API layer. Fourth, choose frameworks based on the bottlenecks and operational surfaces they expose.

The handbook is a good map because it does not stop at "make the model faster." It connects model mechanics to hardware, runtime, scheduling, routing, and operations.

That is the level where inference becomes a real engineering discipline.
