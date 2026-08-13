---
layout: post
title: "LLM Inference Memory Economics"
date: 2026-07-08
tags: [llm, inference, gpu, kv-cache, quantization]
excerpt: "Study notes from the LLM Inference Handbook: why VRAM is the real capacity planner, how model weights and KV cache compete, and where quantization actually helps."
---

Most inference capacity planning starts with the model size. That is a useful first approximation, and it is also where many wrong estimates come from.

The serving question is not "can the weights fit on the GPU?" The real question is "can the weights, KV cache, activations, runtime workspace, batching overhead, and safety margin fit while the service is under load?"

The [LLM Inference Handbook](https://handbook.modular.com/index.md) makes this point across its pages on [GPU memory calculation](https://handbook.modular.com/getting-started/calculating-gpu-memory-for-llms.md), [GPU choice](https://handbook.modular.com/getting-started/choosing-the-right-gpu.md), [quantization](https://handbook.modular.com/model-preparation/llm-quantization.md), and [KV-cache optimization](https://handbook.modular.com/inference-optimization/pagedattention.md). The common thread is that inference is usually constrained by memory before it is constrained by arithmetic.

## Weights are the entry ticket

The simplest memory estimate is model weights:

```text
weight_memory_bytes = parameter_count * bytes_per_parameter
```

A 7B model in FP16 needs roughly:

```text
7B * 2 bytes = 14 GB
```

A 70B model in FP16 needs roughly:

```text
70B * 2 bytes = 140 GB
```

Quantization changes this entry ticket. INT8 cuts weight memory roughly in half relative to FP16. INT4 cuts it again. That is why a model that requires tensor parallelism in FP16 may fit on one high-memory GPU after quantization.

But quantization mostly reduces weight memory. It does not erase the rest of serving memory. If the workload has long contexts or high concurrency, the KV cache can become the larger problem.

## KV cache is the runtime rent

During decode, the model stores key and value vectors for previous tokens so it can avoid recomputing attention over the whole prefix. This is the KV cache. The handbook's [inference mechanics page](https://handbook.modular.com/llm-inference-basics/how-does-llm-inference-work.md#what-is-kv-cache) introduces it at the model level, and the [PagedAttention page](https://handbook.modular.com/inference-optimization/pagedattention.md) explains why serving engines care so much about its allocation.

A rough KV-cache estimate looks like this:

```text
kv_cache_bytes =
  batch_size
  * sequence_length
  * num_layers
  * 2
  * num_kv_heads
  * head_dim
  * bytes_per_element
```

The `2` is for keys and values. The rest is model architecture and active workload shape. The dangerous part is that batch size and sequence length are runtime variables. They grow with concurrent users, chat history, retrieval context, tool traces, and max output length.

Here is a small calculator I would keep around when sizing a deployment:

```python
def gb(x: float) -> float:
    return x / (1024 ** 3)


def weight_memory_gb(params_b: float, bytes_per_param: float) -> float:
    return gb(params_b * 1_000_000_000 * bytes_per_param)


def kv_cache_gb(
    batch_size: int,
    seq_len: int,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    bytes_per_elem: int = 2,
) -> float:
    bytes_total = (
        batch_size
        * seq_len
        * num_layers
        * 2
        * num_kv_heads
        * head_dim
        * bytes_per_elem
    )
    return gb(bytes_total)


weights = weight_memory_gb(params_b=8, bytes_per_param=2)
cache = kv_cache_gb(
    batch_size=64,
    seq_len=8_192,
    num_layers=32,
    num_kv_heads=8,
    head_dim=128,
)

print(f"weights: {weights:.1f} GB")
print(f"KV cache: {cache:.1f} GB")
```

The point is not that this replaces a framework-specific memory profiler. It gives a fast sanity check. If the KV cache estimate is already large before activations and runtime overhead, the deployment is a memory system.

## Quantization changes the feasible architecture

The handbook's [quantization guide](https://handbook.modular.com/model-preparation/llm-quantization.md) separates formats and methods: FP16/BF16, FP8, INT8, INT4, AWQ, SmoothQuant, GPTQ, and related choices. The practical reading is that quantization is a deployment lever, not just a compression trick.

It can change the architecture you are allowed to use.

If a model fits on one GPU after quantization, data parallelism becomes available. That often gives better throughput scaling than splitting one model across GPUs, because each GPU can run an independent copy of the model and serve its own requests. If the model still needs multiple GPUs, tensor or pipeline parallelism becomes necessary, and interconnect cost enters the latency budget.

The trade-off is precision, compatibility, and kernel support. A quantized model is useful only if the runtime can execute it efficiently on the target hardware. The handbook's [GPU selection page](https://handbook.modular.com/getting-started/choosing-the-right-gpu.md) calls out precision support and memory bandwidth as hard constraints. A spec sheet with huge peak FLOPS does not help if the deployment falls back to an unsupported path.

## Memory bandwidth matters during decode

Prefill and decode stress the GPU differently. Prefill has high arithmetic intensity because the model processes many prompt tokens in parallel. Decode generates one token at a time and repeatedly touches weights plus KV cache. That makes memory bandwidth central to user-visible streaming speed.

This is why GPU choice is not only about VRAM capacity. VRAM decides whether the workload fits. Memory bandwidth helps decide whether decode can keep pace. Interconnect decides how painful multi-GPU serving becomes. Framework support decides whether the advertised hardware features are usable from your runtime.

The handbook's [GPU architecture](https://handbook.modular.com/kernel-optimization/gpu-architecture-fundamentals.md) and [GPU memory hierarchy](https://handbook.modular.com/kernel-optimization/gpu-architecture-fundamentals/gpu-memory.md) pages are lower-level than a normal deployment checklist, but they explain the same production fact: moving data is often the cost.

## My study note

I would summarize inference memory with three buckets:

```text
fixed-ish:
  model weights

runtime workload:
  KV cache
  activations
  temporary buffers

operational margin:
  fragmentation
  framework overhead
  safety headroom
  rolling deploy overlap
```

Model weights tell you whether the deployment can start. KV cache tells you how much concurrency and context length the deployment can survive. Quantization reduces the fixed bucket and may change the parallelism strategy. Paged allocation, prefix reuse, and cache-aware routing make the runtime bucket less wasteful.

That is the part I find most important: memory optimization is not only about fitting a bigger model. It is about buying room for useful product behavior: longer context, more concurrent users, lower TTFT, smoother streaming, safer deploys, and fewer emergency scale-ups.

In inference, VRAM is not storage. It is working capital.
