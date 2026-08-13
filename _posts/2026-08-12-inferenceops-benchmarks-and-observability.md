---
layout: post
title: "InferenceOps: Benchmarks, Observability, and Framework Choice"
date: 2026-08-12
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
name: workload_name_here
model: model_under_test
traffic:
  concurrent_users: [low, medium, high]
  arrival_pattern: measured_or_synthetic
tokens:
  prompt_distribution: from_production_trace_or_fixture
  output_distribution: from_production_trace_or_fixture
cache:
  shared_system_prompt: true_or_false
  prefix_reuse_source: trace_or_assumption
slo:
  ttft_p99_ms: product_slo
  e2e_p99_ms: product_slo
```

The benchmark result should answer a deployment question, not produce a trophy number.

The [performance benchmark page](https://handbook.modular.com/inference-optimization/llm-performance-benchmarks.md) lists TTFT, ITL, E2EL, throughput, goodput, memory usage, and cost as benchmark dimensions. A benchmark fixture should therefore include both input shape and acceptance criteria:

```json
{
  "request_id": "fixture_001",
  "prompt_tokens": 0,
  "expected_max_output_tokens": 0,
  "shared_prefix_group": "none",
  "streaming": true,
  "slo": {
    "ttft_ms": 0,
    "e2e_ms": 0
  }
}
```

The zeroes are placeholders. The important part is that the fixture carries the SLO with the request shape. That prevents a benchmark from reporting high TPS while silently violating the product contract.

Here is a minimal comparison function for two benchmark runs:

```python
def percentile(values, pct):
    values = sorted(values)
    index = round((len(values) - 1) * pct / 100)
    return values[index]


def summarize_run(records):
    passed = [
        r for r in records
        if r["ttft_ms"] <= r["slo"]["ttft_ms"]
        and r["e2e_ms"] <= r["slo"]["e2e_ms"]
    ]
    total_tokens = sum(r["prompt_tokens"] + r["output_tokens"] for r in records)
    duration_s = max(r["completed_at_s"] for r in records) - min(
        r["started_at_s"] for r in records
    )
    return {
        "requests": len(records),
        "p99_ttft_ms": percentile([r["ttft_ms"] for r in records], 99),
        "p99_e2e_ms": percentile([r["e2e_ms"] for r in records], 99),
        "tokens_per_second": total_tokens / duration_s,
        "goodput_rps": len(passed) / duration_s,
        "slo_pass_rate": len(passed) / len(records),
    }


def regression_gate(before, after):
    return {
        "goodput_regressed": after["goodput_rps"] < before["goodput_rps"],
        "p99_ttft_regressed": after["p99_ttft_ms"] > before["p99_ttft_ms"],
        "slo_pass_rate_regressed": after["slo_pass_rate"] < before["slo_pass_rate"],
    }
```

This gate encodes the Handbook's goodput framing: useful throughput means completed requests that meet SLOs.

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

The event side matters too. Metrics tell you the distribution. Events explain what changed. A compact JSONL trace can connect a user-visible request to scheduler and runtime state:

```json
{
  "ts": "2026-08-12T00:00:00Z",
  "event": "llm_request_completed",
  "request_id": "req_123",
  "model": "model_under_test",
  "runtime": "runtime_under_test",
  "route": "decode_pool_a",
  "prompt_tokens": 0,
  "output_tokens": 0,
  "ttft_ms": 0,
  "e2e_ms": 0,
  "finish_reason": "stop",
  "prefix_cache_hit": false,
  "kv_cache_used_bytes": 0,
  "worker_active_sequences": 0
}
```

The Handbook's [observability page](https://handbook.modular.com/infrastructure-and-operations/comprehensive-observability.md) talks about metrics, logs, and events across infrastructure, application, and model layers. This trace shape is just that idea made concrete.

For debugging, I would keep a few derived checks close to the metrics:

```python
def explain_ttft_change(before, after):
    candidates = []
    if after["prompt_tokens_p95"] > before["prompt_tokens_p95"]:
        candidates.append("prompt length increased")
    if after["queue_ms_p99"] > before["queue_ms_p99"]:
        candidates.append("queueing increased")
    if after["prefix_cache_hit_rate"] < before["prefix_cache_hit_rate"]:
        candidates.append("prefix cache hit rate dropped")
    if after["kv_cache_used_ratio_p99"] > before["kv_cache_used_ratio_p99"]:
        candidates.append("KV cache pressure increased")
    return candidates
```

The function does not diagnose by itself. It keeps the investigation tied to measured surfaces.

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

The [framework-choice page](https://handbook.modular.com/getting-started/choosing-the-right-inference-framework.md#library-mode-vs-server-mode) separates library mode and server mode. I would turn that into an explicit decision record:

```yaml
decision: inference_runtime_mode
options:
  library_mode:
    choose_when:
      - application needs direct control over request lifecycle
      - custom integration matters more than serving isolation
  server_mode:
    choose_when:
      - standardized API boundary matters
      - runtime should scale independently
      - operations team needs separate deployment and rollback
selected: server_mode
reason:
  - deployment requires independent scaling of inference workers
  - gateway can remain stable while runtime changes
```

The selected option above is an example record, not a universal recommendation. The useful habit is to write down which operational surface drove the choice.

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

Safe rollout can also be encoded. The [InferenceOps page](https://handbook.modular.com/infrastructure-and-operations/inferenceops-and-management.md) covers standardized deployment workflows, safe updates, centralized management, and cost control. A canary gate can check those concerns before shifting traffic:

```python
def canary_gate(baseline, canary):
    checks = {
        "error_rate_ok": canary["error_rate"] <= baseline["error_rate"],
        "p99_ttft_ok": canary["p99_ttft_ms"] <= baseline["p99_ttft_ms"],
        "goodput_ok": canary["goodput_rps"] >= baseline["goodput_rps"],
        "cost_ok": canary["cost_per_1k_tokens"] <= baseline["cost_per_1k_tokens"],
    }
    return all(checks.values()), checks
```

This kind of gate keeps deployment from becoming a vibes-based decision. The exact thresholds belong to the product and infra team. The measured dimensions come from the Handbook's benchmark and operations framing.

## My study note

The production view of inference can be compressed into one rule:

```text
Optimize against SLO-qualified workload behavior, not abstract throughput.
```

That rule pushes you toward four habits. First, define the workload shape before benchmarking. Second, measure goodput and P99 latency, not only TPS. Third, instrument the scheduler and KV cache, because many failures hide below the API layer. Fourth, choose frameworks based on the bottlenecks and operational surfaces they expose.

The handbook is a good map because it does not stop at "make the model faster." It connects model mechanics to hardware, runtime, scheduling, routing, and operations.

That is the level where inference becomes a real engineering discipline.
