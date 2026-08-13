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

The Handbook's [GPU memory page](https://handbook.modular.com/getting-started/calculating-gpu-memory-for-llms.md) gives the quick sizing form:

```text
Memory (GB) = P * (Q / 8) * (1 + Overhead)
```

`P` is parameter count in billions. `Q` is precision in bits. The overhead term covers runtime memory beyond weights: KV cache, activations, workspace buffers, framework allocations, CUDA graphs, and safety headroom. The Handbook also warns that this percentage shortcut is not an exact capacity model because KV cache depends on sequence length, batch size, concurrency, layers, and hidden dimensions.

A 7B model in FP16 needs roughly:

```text
7B * 2 bytes = 14 GB
```

A 70B model in FP16 needs roughly:

```text
70B * 2 bytes = 140 GB
```

Quantization changes this entry ticket. INT8 cuts weight memory roughly in half relative to FP16. INT4 cuts it again. That is why a model that requires tensor parallelism in FP16 may fit on one high-memory GPU after quantization.

The Handbook's [quantization page](https://handbook.modular.com/model-preparation/llm-quantization.md) separates weight memory from serving memory. Weight quantization reduces how many bytes the model weights require. It does not reduce KV-cache memory per token unless the KV cache is quantized separately.

I would encode the weight side as a small table rather than a sentence:

```python
BITS_PER_FORMAT = {
    "fp32": 32,
    "bf16": 16,
    "fp16": 16,
    "fp8": 8,
    "int8": 8,
    "int4": 4,
}


def weight_memory_gb(params_b: float, fmt: str) -> float:
    bits = BITS_PER_FORMAT[fmt]
    return params_b * (bits / 8)


for fmt in ["fp32", "fp16", "int8", "int4"]:
    print(fmt, weight_memory_gb(params_b=7, fmt=fmt))

# fp32 28.0
# fp16 14.0
# int8 7.0
# int4 3.5
```

Those numbers only describe weights. If the workload has long contexts or high concurrency, the KV cache can become the larger problem.

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
from dataclasses import dataclass


def gb(x: float) -> float:
    return x / (1024 ** 3)


@dataclass(frozen=True)
class ModelShape:
    params_b: float
    num_layers: int
    num_kv_heads: int
    head_dim: int
    weight_bits: int = 16
    kv_cache_bits: int = 16


def weight_memory_from_shape_gb(model: ModelShape) -> float:
    return model.params_b * (model.weight_bits / 8)


def kv_cache_gb(
    model: ModelShape,
    batch_size: int,
    seq_len: int,
) -> float:
    bytes_per_elem = model.kv_cache_bits / 8
    bytes_total = (
        batch_size
        * seq_len
        * model.num_layers
        * 2
        * model.num_kv_heads
        * model.head_dim
        * bytes_per_elem
    )
    return gb(bytes_total)


model = ModelShape(
    params_b=8,
    num_layers=32,
    num_kv_heads=8,
    head_dim=128,
)

weights = weight_memory_from_shape_gb(model)
cache = kv_cache_gb(
    model,
    batch_size=64,
    seq_len=8_192,
)

print(f"weights: {weights:.1f} GB")
print(f"KV cache: {cache:.1f} GB")
```

The point is not that this replaces a framework-specific memory profiler. It gives a fast sanity check. If the KV cache estimate is already large before activations and runtime overhead, the deployment is a memory system.

The `num_kv_heads` field matters. Many modern models use grouped-query or multi-query attention, where the number of KV heads can be smaller than the number of query heads. The KV-cache formula cares about stored key/value heads, not just model parameter count. This is the kind of detail that gets lost when planning stops at "7B", "14B", or "70B".

The same calculator can show sensitivity without pretending to benchmark anything:

```python
def cache_sweep(model: ModelShape, batch_sizes, seq_lens):
    rows = []
    for batch in batch_sizes:
        for seq_len in seq_lens:
            rows.append({
                "batch": batch,
                "seq_len": seq_len,
                "kv_cache_gb": round(kv_cache_gb(model, batch, seq_len), 2),
            })
    return rows


for row in cache_sweep(model, batch_sizes=[8, 32, 64], seq_lens=[2048, 8192, 32768]):
    print(row)
```

The output is deployment-specific because the model shape is deployment-specific. The lesson is stable: KV cache grows linearly with active sequences and context length.

## Quantization changes the feasible architecture

The handbook's [quantization guide](https://handbook.modular.com/model-preparation/llm-quantization.md) separates formats and methods: FP16/BF16, FP8, INT8, INT4, AWQ, SmoothQuant, GPTQ, and related choices. The practical reading is that quantization is a deployment lever, not just a compression trick.

It can change the architecture you are allowed to use.

If a model fits on one GPU after quantization, data parallelism becomes available. That often gives better throughput scaling than splitting one model across GPUs, because each GPU can run an independent copy of the model and serve its own requests. If the model still needs multiple GPUs, tensor or pipeline parallelism becomes necessary, and interconnect cost enters the latency budget.

The trade-off is precision, compatibility, and kernel support. A quantized model is useful only if the runtime can execute it efficiently on the target hardware. The handbook's [GPU selection page](https://handbook.modular.com/getting-started/choosing-the-right-gpu.md) calls out precision support and memory bandwidth as hard constraints. A spec sheet with huge peak FLOPS does not help if the deployment falls back to an unsupported path.

This is also where capacity planning should include hardware capability checks:

```python
def validate_precision_support(gpu, model):
    if model.weight_bits == 8 and "fp8" not in gpu["native_precisions"]:
        return "FP8 weights need native FP8 support or an efficient fallback path."
    if model.weight_bits == 4 and not gpu["runtime_supports_int4"]:
        return "INT4 weight storage is not enough; the runtime needs optimized kernels."
    return "precision path looks compatible"


gpu = {
    "name": "target-gpu",
    "native_precisions": {"fp16", "bf16", "int8"},
    "runtime_supports_int4": False,
}
```

The strings are policy checks, not hardware claims. The claims come from the Handbook: precision support and runtime/kernel support decide whether lower-bit formats turn into real serving capacity.

## Memory bandwidth matters during decode

Prefill and decode stress the GPU differently. Prefill has high arithmetic intensity because the model processes many prompt tokens in parallel. Decode generates one token at a time and repeatedly touches weights plus KV cache. That makes memory bandwidth central to user-visible streaming speed.

This is why GPU choice is not only about VRAM capacity. VRAM decides whether the workload fits. Memory bandwidth helps decide whether decode can keep pace. Interconnect decides how painful multi-GPU serving becomes. Framework support decides whether the advertised hardware features are usable from your runtime.

The handbook's [GPU architecture](https://handbook.modular.com/kernel-optimization/gpu-architecture-fundamentals.md) and [GPU memory hierarchy](https://handbook.modular.com/kernel-optimization/gpu-architecture-fundamentals/gpu-memory.md) pages are lower-level than a normal deployment checklist, but they explain the same production fact: moving data is often the cost.

The [GPU choice page](https://handbook.modular.com/getting-started/choosing-the-right-gpu.md#memory-bandwidth) gives the bandwidth-bound intuition for single-stream decode:

```text
maximum decode tokens/sec ~= memory bandwidth / bytes read per token
```

A conservative estimator can keep the assumption visible:

```python
def bandwidth_bound_decode_tps(memory_bandwidth_gb_s, bytes_read_per_token_gb):
    return memory_bandwidth_gb_s / bytes_read_per_token_gb


def bytes_read_per_token_gb(params_b, weight_bits):
    return params_b * (weight_bits / 8)


read_gb = bytes_read_per_token_gb(params_b=70, weight_bits=16)
bound = bandwidth_bound_decode_tps(memory_bandwidth_gb_s=3350, bytes_read_per_token_gb=read_gb)
print(round(bound, 1))
```

This estimator intentionally ignores KV-cache traffic and runtime overhead. Its job is to show why decode can be bandwidth-bound even when the GPU has high peak compute throughput.

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
