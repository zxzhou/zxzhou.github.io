---
layout: post
title: "Distributed LLM Inference Is a Routing Problem"
date: 2026-08-13
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

## Routing answers where the request should go

Normal load balancing often starts with round-robin or least-loaded selection. LLM inference needs more context.

A request may share a long prefix with state already resident on one worker. Sending it elsewhere recomputes prefill and misses cache locality. A worker may look lightly loaded by request count but have high KV-cache pressure. A decode-heavy worker may produce poor streaming cadence if it also receives a long prefill. A replica in another region may have spare capacity but add network latency.

The handbook's [inference routing](https://handbook.modular.com/inference-optimization/inference-routing.md) page lists strategies such as least-loaded routing, KV-cache utilization-aware routing, prefix-aware routing, and prefill/decode-aware routing. The practical version is usually a hybrid score:

```python
def route_score(worker, request):
    return (
        0.35 * worker.queue_pressure()
        + 0.25 * worker.kv_cache_pressure()
        + 0.20 * worker.prefill_decode_imbalance(request)
        + 0.15 * worker.network_latency_ms(request.region)
        - 0.30 * worker.prefix_cache_hit_score(request.prompt_prefix_hash)
    )


def choose_worker(workers, request):
    candidates = [w for w in workers if w.can_serve(request.model)]
    healthy = [w for w in candidates if w.is_healthy()]
    return min(healthy, key=lambda w: route_score(w, request))
```

The weights are made up. The shape is the useful part. Routing is not only spreading requests; it is preserving scarce state.

## Prefill and decode want different machines

Prefill and decode have different resource profiles. Prefill processes the prompt and tends to benefit from high compute throughput. Decode produces one token at a time and repeatedly touches KV cache, so memory bandwidth and cache residency matter.

[Prefill-decode disaggregation](https://handbook.modular.com/inference-optimization/prefill-decode-disaggregation.md) splits those phases across different workers. A prefill worker computes the prompt state. A decode worker continues generation. This can improve utilization because each pool is tuned for its phase.

The trade-off is KV-cache movement. If the prefill worker and decode worker are separate, the system must transfer the cache or have a mechanism for the decode worker to access it. Across nodes or clusters, that transfer can become expensive enough to erase the benefit.

This is the pattern behind many distributed inference decisions. A clean architecture diagram can hide the cost of moving state. The hard question is not only "can I split this work?" It is "what state moves, how often, and over which link?"

## Multi-region inference adds product constraints

The handbook's [distributed inference](https://handbook.modular.com/infrastructure-and-operations/distributed-inference.md) page expands the problem beyond one node. Multi-region serving can reduce user latency, improve fault tolerance, and use cheaper capacity. It also creates consistency, routing, observability, and cost-accounting problems.

For LLMs, state is heavier than in many stateless web services. Model weights are large. KV cache is request-specific and grows with context length. Prompt prefixes may be reused locally. Warm capacity matters. Cold starts hurt.

That means a global inference layer has to understand more than HTTP health checks. It needs model placement, runtime compatibility, live capacity, cache pressure, and SLO status. Otherwise, it can make locally rational decisions that create globally worse latency or cost.

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

The strongest inference routers are state-aware. They know where model replicas live, where prefixes are cached, which workers are saturated, which pools are serving prefill versus decode, and which requests are close to violating an SLO.

That is the difference between a model-serving cluster and an inference system. A cluster has GPUs. An inference system has a policy for scarce state.
