---
layout: post
title: "Distributed LLM Inference Is a Routing Problem"
date: 2026-07-29
tags: [llm, inference, distributed-systems, routing, gpu]
excerpt: "Study notes from the LLM Inference Handbook: data, tensor, pipeline, and expert parallelism matter, but the production system lives or dies by routing requests to the right scarce state."
---

Distributed inference usually gets described as a parallelism problem. That is true at the model-execution layer, but incomplete at the serving layer.

Once a deployment has multiple GPUs, nodes, regions, models, or runtimes, inference becomes a routing problem. The system must decide where each request should go while considering queue depth, model placement, KV cache locality, prefill pressure, decode pressure, GPU memory, failure domains, and latency.

The [LLM Inference Handbook](https://handbook.modular.com/index.md) lays the pieces out across [parallelism strategies](https://handbook.modular.com/inference-optimization/data-tensor-pipeline-expert-hybrid-parallelism.md), [distributed inference](https://handbook.modular.com/infrastructure-and-operations/distributed-inference.md), [prefill-decode disaggregation](https://handbook.modular.com/inference-optimization/prefill-decode-disaggregation.md), and [inference routing](https://handbook.modular.com/inference-optimization/inference-routing.md). Read together, they suggest a useful rule: parallelism lets the model run; routing decides whether the service behaves well.

## Parallelism answers where the model lives

The handbook separates several parallelism modes:

| Strategy | What it splits | Why you use it |
|---|---|---|
| Data parallelism | Requests across model replicas | Increase throughput when the full model fits per worker |
| Tensor parallelism | Tensor operations across GPUs | Serve models that are too large or too slow for one GPU |
| Pipeline parallelism | Model layers across stages | Spread a deep model across devices |
| Expert parallelism | MoE experts across devices | Route tokens to specialized experts |
| Hybrid parallelism | Multiple dimensions together | Serve large models at scale |

Data parallelism is the cleanest serving path when it is available. Each worker holds a full copy of the model and handles different requests. The cost is duplicated weights and separate KV caches. The benefit is simple scaling and fewer per-token communication costs.

Tensor and pipeline parallelism solve the "model does not fit" problem, but introduce communication overhead. Now the speed of interconnects matters. A deployment may gain memory capacity while losing some expected throughput because every step needs coordination across devices.

This is why quantization can have a second-order effect. If quantization makes a model fit on one GPU, it may move the system from tensor parallelism back to data parallelism. That can simplify routing and improve throughput.

The first artifact I would create is a placement table. The [distributed inference](https://handbook.modular.com/infrastructure-and-operations/distributed-inference.md) page separates model scale, reliability, cost, network overhead, state management, and observability. A placement table makes those constraints explicit before the router sees traffic:

```yaml
models:
  small_chat_model:
    runtime: runtime_under_test
    precision: precision_under_test
    placement:
      - region: us-west
        replicas: n
        parallelism: data
      - region: us-east
        replicas: n
        parallelism: data

  large_chat_model:
    runtime: runtime_under_test
    precision: precision_under_test
    placement:
      - region: us-west
        replicas: n
        parallelism: tensor
        tensor_parallel_size: n
```

This template does not claim which runtime is best. It records where each model can actually run, how it is split, and what the router may choose.

## Routing answers where the request should go

Normal load balancing often starts with round-robin or least-loaded selection. LLM inference needs more context.

A request may share a long prefix with state already resident on one worker. Sending it elsewhere recomputes prefill and misses cache locality. A worker may look lightly loaded by request count but have high KV-cache pressure. A decode-heavy worker may produce poor streaming cadence if it also receives a long prefill. A replica in another region may have spare capacity but add network latency.

The handbook's [inference routing](https://handbook.modular.com/inference-optimization/inference-routing.md) page lists round-robin, random, least-loaded, direct routing, KV-cache utilization-aware routing, prefix-aware routing, and prefill/decode-aware routing. A production router usually turns those into eligibility checks followed by a policy choice:

```python
def eligible(worker, request):
    return (
        worker.healthy
        and request.model in worker.models
        and worker.free_kv_blocks >= request.estimated_kv_blocks
        and worker.region in request.allowed_regions
    )


def choose_worker(workers, request):
    candidates = [w for w in workers if eligible(w, request)]
    if not candidates:
        raise RuntimeError("no eligible inference worker")

    prefix_hits = [
        w for w in candidates
        if request.prompt_prefix_hash in w.prefix_cache_keys
    ]
    if prefix_hits:
        return min(prefix_hits, key=lambda w: w.queue_depth)

    return min(
        candidates,
        key=lambda w: (
            w.queue_depth,
            w.kv_cache_used_ratio,
            w.network_latency_ms[request.client_region],
        ),
    )
```

This policy encodes the Handbook concepts without pretending there is a universal formula. Eligibility protects correctness and capacity. Prefix-aware routing preserves reusable state. Queue and KV pressure avoid sending work to a worker that is already saturated.

## Prefill and decode want different machines

Prefill and decode have different resource profiles. Prefill processes the prompt and tends to benefit from high compute throughput. Decode produces one token at a time and repeatedly touches KV cache, so memory bandwidth and cache residency matter.

[Prefill-decode disaggregation](https://handbook.modular.com/inference-optimization/prefill-decode-disaggregation.md) splits those phases across different workers. A prefill worker computes the prompt state. A decode worker continues generation. This can improve utilization because each pool is tuned for its phase.

The trade-off is KV-cache movement. If the prefill worker and decode worker are separate, the system must transfer the cache or have a mechanism for the decode worker to access it. Across nodes or clusters, that transfer can become expensive enough to erase the benefit.

This is the pattern behind many distributed inference decisions. A clean architecture diagram can hide the cost of moving state. The hard question is not only "can I split this work?" It is "what state moves, how often, and over which link?"

The Handbook's [prefill-decode disaggregation page](https://handbook.modular.com/inference-optimization/prefill-decode-disaggregation.md#the-core-problem-kv-cache-movement) names KV-cache movement as the core cross-cluster problem. You can estimate the transfer size with the same cache shape used for memory planning:

```python
def kv_transfer_bytes(
    prompt_tokens,
    num_layers,
    num_kv_heads,
    head_dim,
    bytes_per_element=2,
):
    return (
        prompt_tokens
        * num_layers
        * 2
        * num_kv_heads
        * head_dim
        * bytes_per_element
    )


def transfer_ms(bytes_to_move, link_gb_per_s):
    return (bytes_to_move / (link_gb_per_s * 1024 ** 3)) * 1000
```

This code does not decide whether disaggregation is good. It shows which number has to be paid when prefill and decode are separated.

The pool design also needs to keep phase pressure visible:

```yaml
pools:
  prefill:
    optimized_for: prompt_processing
    routing_signals:
      - queued_prefill_tokens
      - available_compute
      - kv_transfer_target

  decode:
    optimized_for: token_streaming
    routing_signals:
      - active_sequences
      - kv_cache_used_ratio
      - p99_itl_ms
```

The [routing page](https://handbook.modular.com/inference-optimization/inference-routing.md#prefilldecode-aware-routing) supports this split directly: prefill/decode-aware routing treats those phases as different scheduling pressures.

## Multi-region inference adds product constraints

The handbook's [distributed inference](https://handbook.modular.com/infrastructure-and-operations/distributed-inference.md) page expands the problem beyond one node. Multi-region serving can reduce user latency, improve fault tolerance, and use cheaper capacity. It also creates consistency, routing, observability, and cost-accounting problems.

For LLMs, state is heavier than in many stateless web services. Model weights are large. KV cache is request-specific and grows with context length. Prompt prefixes may be reused locally. Warm capacity matters. Cold starts hurt.

That means a global inference layer has to understand more than HTTP health checks. It needs model placement, runtime compatibility, live capacity, cache pressure, and SLO status. Otherwise, it can make locally rational decisions that create globally worse latency or cost.

A region-level router can keep those signals separate:

```python
def choose_region(regions, request):
    candidates = [
        r for r in regions
        if request.model in r.models
        and r.status == "healthy"
        and r.capacity.has_headroom_for(request)
    ]
    return min(
        candidates,
        key=lambda r: (
            r.client_latency_ms[request.client_region],
            r.slo_burn_rate[request.model],
            r.estimated_cost_per_1k_tokens[request.model],
        ),
    )
```

Latency, SLO burn, and cost are different objectives. Combining them in code forces the operator to decide the order of priority instead of hiding it inside "least loaded."

## My study note

I would separate distributed inference into three questions:

```text
placement:
  Where do the model weights live?
  Which runtimes and precisions are supported there?

parallelism:
  Does one request run on one worker, many GPUs, many nodes, or many experts?

routing:
  Which worker should receive this specific request right now?
```

Most tutorials spend time on the second question. Production systems spend much of their pain on the first and third.

A useful inference router is state-aware. It knows where model replicas live, where prefixes are cached, which workers are saturated, which pools are serving prefill versus decode, and which requests are close to violating an SLO.

That is the difference between a model-serving cluster and an inference system. A cluster has GPUs. An inference system has a policy for scarce state.
