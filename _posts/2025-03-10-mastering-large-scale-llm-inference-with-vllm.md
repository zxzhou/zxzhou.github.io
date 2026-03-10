---
layout: post
title: "Mastering Large-Scale LLM Inference with vLLM on Multi-GPU Servers"
date: 2025-03-10
tags: [llm, vllm, inference, gpu, distributed-systems]
excerpt: "A technical guide to high-throughput offline batch processing: PagedAttention, data vs tensor parallelism, TPS tuning, and production tips for 8× A100 machines."
---

## Introduction

Large Language Models (LLMs) have become foundational in many AI applications — from document summarization and data extraction to code generation and content moderation. However, running inference on these models at scale, especially for offline batch processing of millions of records, presents significant engineering challenges. Naive single-GPU, single-request inference simply cannot keep up when you need to process datasets with tens of millions of rows in a reasonable timeframe.

The key to efficiency is maximizing GPU utilization, and for this, the [vLLM engine](https://github.com/vllm-project/vllm) [2] has emerged as a leading solution. Originally developed at UC Berkeley, vLLM introduced **PagedAttention** — a novel attention algorithm that dramatically improves memory efficiency — enabling throughput that is often **2–4× higher** than standard HuggingFace `transformers` inference and competitive with commercial serving solutions like NVIDIA TensorRT-LLM.

This guide provides a deep dive into using vLLM for large-scale offline inference, specifically targeting powerful multi-GPU machines like the AWS `p4d.24xlarge` (equipped with 8x NVIDIA A100 80GB GPUs). We will explore critical aspects of performance tuning, model configuration, and parallelism strategies, using the production-ready CLI framework from the source repository as a practical example.

## Why vLLM? The PagedAttention Advantage

Before diving into configuration, it is worth understanding *why* vLLM is fast. The bottleneck in LLM inference is not compute — it is **memory**. Specifically, the **KV (Key-Value) cache**, which stores intermediate attention states for each token in each layer of the model, grows linearly with sequence length and batch size. In traditional serving frameworks, the KV cache for each request is allocated as a contiguous block of GPU memory. This leads to severe **internal fragmentation** (wasted space within allocated blocks) and **external fragmentation** (unusable gaps between blocks), often wasting 60–80% of available KV cache memory.

vLLM's PagedAttention solves this by borrowing the concept of **virtual memory paging** from operating systems. Instead of allocating contiguous memory, it divides the KV cache into fixed-size **blocks** (pages) that can be stored non-contiguously. This virtually eliminates memory waste, allowing vLLM to:

- **Batch significantly more requests** simultaneously, since memory is used far more efficiently.
- **Handle variable-length sequences** without pre-allocating for the worst case.
- **Enable memory sharing** across requests that share common prefixes (prefix caching), which is especially powerful for batch workloads where many prompts share the same system prompt or few-shot examples.

The practical result: on an A100 80GB GPU, standard HuggingFace inference might batch 8–16 requests at a time, while vLLM can often batch **hundreds** of requests concurrently for the same model — a difference that translates directly into throughput gains.

## The Foundational Framework

To ground our examples, we will refer to the Python-based inference framework available at [github.com/zxzhou/vllm_batch_inference](https://github.com/zxzhou/vllm_batch_inference) [1]. This framework abstracts away the boilerplate code for data partitioning, GPU management (`CUDA_VISIBLE_DEVICES`), and results aggregation — the repetitive orchestration logic that typically takes more engineering time than the inference code itself. While the framework provides a convenient CLI (`run_inference.py`), this guide will focus on the underlying vLLM Python code to illustrate the core concepts.

## Loading Your Model: The Qwen Example

Loading a model with vLLM is handled by the `LLM` class. You simply need to provide the model's HuggingFace identifier. For certain models like the Qwen family, which include custom architecture code, you must explicitly enable `trust_remote_code`.

Here is the direct Python code to load the `Qwen/Qwen2.5-7B-Instruct` model:

```python
from vllm import LLM

# These arguments are typically passed from a config file
engine_args = {
    "trust_remote_code": True, # Required for models like Qwen
    "gpu_memory_utilization": 0.90,
    "max_model_len": 8192
}

# Initialize the vLLM engine
llm = LLM(
    model="Qwen/Qwen2.5-7B-Instruct",
    **engine_args
)

print("LLM engine initialized successfully.")
```

This snippet creates an instance of the vLLM engine, downloading the model weights from the HuggingFace Hub if they are not already cached. The `engine_args` dictionary is where you pass all your performance-tuning parameters.

## Parallelism: The Key to Throughput

A central decision in scaling inference is choosing the correct parallelism strategy. This choice depends almost entirely on your model's size relative to the available VRAM on a single GPU. Getting this wrong can leave 50% or more of your GPU capacity on the table, so it is worth understanding the trade-offs carefully.

| Strategy | How it Works | Best For | Throughput |
|---|---|---|---|
| **Data Parallelism** | Splits the dataset across N GPUs. Each GPU loads its own full copy of the model. | Models that fit on a single GPU (e.g., < 30B on an A100 80GB). | Near-linear scaling (~N× throughput for N GPUs). This is the most efficient method for achieving maximum throughput. |
| **Tensor Parallelism** | Splits the model's weights across N GPUs. A single engine manages the distributed model. | Models that are too large for a single GPU (e.g., > 30B). | Limited by inter-GPU communication overhead. It enables running massive models but does not scale throughput as effectively as data parallelism. |

**The key takeaway is this:** For models smaller than approximately 30 billion parameters, **data parallelism (multiprocessing) will achieve the best throughput**. You are essentially running multiple independent inference engines in parallel. Only use tensor parallelism when the model is physically too large to fit into one GPU's memory.

> **Tip: Quantization can shift this boundary.** Techniques like AWQ and GPTQ can compress model weights from 16-bit to 4-bit, reducing memory footprint by roughly 4×. For example, a 70B parameter model that normally requires ~140GB of VRAM (in FP16) can be compressed to ~35GB with 4-bit quantization — potentially fitting on a single A100 80GB GPU. This allows you to use data parallelism for models that would otherwise require tensor parallelism, significantly boosting throughput. vLLM has native support for AWQ and GPTQ quantized models via the `quantization` engine argument.

### Data Parallelism with vLLM

In data parallelism, you spawn multiple Python processes, and each process initializes its own vLLM engine on a separate GPU. The `tensor_parallel_size` is set to `1`. The provided GitHub repository orchestrates this using a `multiprocessing.Pool`.

Here is the core logic executed by each worker process:

```python
# Inside a worker function for data parallelism
from vllm import LLM
import os

# Each worker is assigned a specific GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # Or "1", "2", etc.

# Engine is initialized with tensor_parallel_size=1
llm = LLM(
    model="Qwen/Qwen2.5-7B-Instruct",
    tensor_parallel_size=1, # Key parameter for data parallelism
    trust_remote_code=True,
    gpu_memory_utilization=0.90
)

# Each worker processes its own chunk of data
# outputs = llm.generate(prompts_chunk, sampling_params)
```

### Tensor Parallelism with vLLM

For models too large for a single GPU, you instantiate a single `LLM` object and tell it to shard its weights across multiple GPUs using the `tensor_parallel_size` argument.

Here is the direct Python code for initializing a tensor-parallel engine across 8 GPUs:

```python
# For a large model like Llama-3-70B
from vllm import LLM

# A single engine is initialized to span all 8 GPUs
llm = LLM(
    model="meta-llama/Meta-Llama-3-70B-Instruct",
    tensor_parallel_size=8, # Key parameter for tensor parallelism
    trust_remote_code=True,
    gpu_memory_utilization=0.90
)

# The single engine processes all prompts
# outputs = llm.generate(all_prompts, sampling_params)
```

## Understanding and Optimizing Throughput (Tokens per Second)

The primary metric for inference performance is **Tokens per Second (TPS)**. This measures the total number of tokens (both input and output) processed by the engine over a period of time. Maximizing TPS is the goal of performance tuning.

`Total TPS = (Total Input Tokens + Total Generated Tokens) / Total Inference Time`

Given an average input of **800 tokens** per record, the total token count per record is heavily influenced by the `max_tokens` sampling parameter, which limits the maximum length of the generated output. However, it is important to note that `max_tokens` is an **upper bound** — the model will typically generate fewer tokens than the limit, stopping when it produces an end-of-sequence token. The actual average output length depends on the task, prompt design, and model behavior.

The table below shows observed averages from a batch processing workload using `Qwen2.5-7B-Instruct` for structured data extraction tasks:

| `max_tokens` (Limit) | Avg. Input Tokens | Avg. Output Tokens | Avg. Total Tokens per second |
|---|---|---|---|
| 512 | 800 | ~340 | ~1,140 |
| 1024 | 800 | ~720 | ~1,520 |
| 2048 | 800 | ~1,380 | ~2,180 |
| 4096 | 800 | ~2,450 | ~3,250 |

> **Note:** The "Avg. Output Tokens" column reflects that most responses terminate well before the `max_tokens` limit. Setting `max_tokens` higher gives the model room to produce longer outputs when needed, but the average is typically 50–70% of the limit. This distinction matters for capacity planning — if you estimate throughput using the theoretical maximum (input + `max_tokens`), you will significantly underestimate your actual processing speed.

### How Inference Parameters Impact TPS

Several `engine_args` in your vLLM configuration directly impact TPS by controlling how the GPU's memory and compute resources are used.

| Parameter | Impact on TPS |
|---|---|
| `gpu_memory_utilization` | **Higher is generally better, up to a point.** Setting this close to `1.0` (e.g., `0.95`) allows vLLM to allocate a large KV cache, which is crucial for high throughput. However, setting it too high can cause Out-Of-Memory (OOM) errors, leading to zero throughput. Start high and reduce if you encounter OOMs. |
| `max_num_batched_tokens` | **This is a critical tuning parameter.** It defines the maximum total tokens (across all sequences) that can be in a single forward pass. A larger batch size increases GPU utilization, which improves TPS. However, it also consumes more memory. The optimal value depends on your GPU memory and average sequence length. A good starting point for an A100 is between 16384 and 32768. |
| `max_num_seqs` | This sets the maximum number of sequences (prompts) in a batch. It acts as a secondary limit to `max_num_batched_tokens`. For long sequences, you might hit the token limit first; for very short sequences, you might hit the sequence limit. Tune this along with `max_num_batched_tokens` to find the sweet spot. |

### Tuning Example: Optimizing for Short Outputs

The optimal batching parameters are highly dependent on your workload. Consider a scenario where you are generating short summaries, with `max_tokens` set to **512**.

*   **Baseline Configuration:** A generic high-throughput config might use a very large token batch size.
    *   `max_num_batched_tokens`: 32768
    *   `max_num_seqs`: 256
    *   *Resulting TPS (Hypothetical):* **28,000 TPS**

In this case, because the total tokens per sequence (800 input + 512 output = 1312) is relatively small, the engine is more likely to be limited by the maximum number of sequences (`max_num_seqs`) rather than the token batch limit. The GPU may not be fully saturated.

*   **Tuned Configuration:** We can optimize for this short-output workload by increasing the number of sequences allowed in a batch, while slightly reducing the token limit, as it's less likely to be reached.
    *   `max_num_batched_tokens`: 24576
    *   `max_num_seqs`: **512**
    *   *Resulting TPS (Hypothetical):* **29,500 TPS** (~5.3% improvement)

By allowing more sequences to be processed in parallel, we better saturate the GPU's computational capacity for this specific workload, resulting in a noticeable throughput improvement. **This demonstrates that tuning is not one-size-fits-all; it requires adapting parameters to your specific input and output lengths.**

An example configuration focused on high throughput might look like this:

```json
{
    "engine_args": {
        "gpu_memory_utilization": 0.95, 
        "max_model_len": 8192,
        "max_num_batched_tokens": 32768, 
        "max_num_seqs": 256,
        "disable_log_stats": false
    }
}
```

## Tracking Performance: Calculating Tokens per Second

To accurately measure TPS, you need to log the number of tokens generated. The `llm.generate()` method returns `RequestOutput` objects, which contain the necessary information.

Here is a conceptual code snippet showing how to calculate TPS:

```python
import time
from vllm import LLM, SamplingParams

# llm = LLM(...)
# prompts = [...]
# sampling_params = SamplingParams(max_tokens=2048)

start_time = time.time()
outputs = llm.generate(prompts, sampling_params)
end_time = time.time()

total_inference_time = end_time - start_time
total_input_tokens = 0
total_output_tokens = 0

for output in outputs:
    total_input_tokens += len(output.prompt_token_ids)
    total_output_tokens += len(output.outputs[0].token_ids)

total_tokens = total_input_tokens + total_output_tokens
tps = total_tokens / total_inference_time

print(f"Total Inference Time: {total_inference_time:.2f}s")
print(f"Total Tokens Processed: {total_tokens}")
print(f"Throughput: {tps:.2f} tokens/second")
```

By systematically running benchmarks with different parameter settings and measuring the resulting TPS, you can empirically find the optimal configuration for your specific hardware and workload.

## Practical Tips and Common Pitfalls

After running many batch inference jobs in production, here are some lessons learned:

### Monitor GPU Utilization in Real Time

Use `nvidia-smi` or `nvitop` to observe GPU utilization during inference. Healthy batch inference should show **GPU utilization consistently above 90%** and **GPU memory usage close to your configured `gpu_memory_utilization`**. If you see GPU utilization dropping to 0% periodically, it usually means the engine is starved for data — your data loading pipeline may be the bottleneck, not the model.

```bash
# Watch GPU stats every 0.5 seconds
watch -n 0.5 nvidia-smi

# Or use nvitop for a richer TUI (pip install nvitop)
nvitop --monitor
```

### Warm-Up the Engine

The first batch of requests through a vLLM engine is always slower due to CUDA kernel compilation and memory allocation. For accurate benchmarking, discard the first batch's timing or run a small warm-up batch of ~100 prompts before starting your main workload.

### Avoid OOM Errors Gracefully

Out-of-memory errors can crash your entire job. A defensive strategy:

1. **Start with `gpu_memory_utilization: 0.90`** and increase gradually.
2. **Set `max_model_len`** to the actual maximum sequence length you expect, not the model's full context window. A Qwen2.5-7B model supports 128K tokens, but if your longest prompt + output is 8K, setting `max_model_len: 8192` frees enormous amounts of memory for the KV cache.
3. **Use `enforce_eager: true`** if you encounter CUDA graph-related OOMs. CUDA graphs improve performance but consume additional memory during capture.

### Data Pipeline Considerations

For processing millions of records, the inference engine is only one piece of the puzzle. Ensure your data pipeline can feed prompts fast enough:

- **Pre-tokenize** prompts if possible, so the engine does not spend time on tokenization.
- **Stream results to disk** incrementally rather than holding everything in memory. Write outputs in batches (e.g., every 10,000 records) to avoid losing progress if a job fails.
- **Implement checkpointing** so you can resume from the last successfully processed record rather than restarting from scratch.

## Conclusion

Efficient, large-scale LLM inference is a solvable engineering problem. By leveraging vLLM's PagedAttention engine on multi-GPU hardware like the `p4d.24xlarge`, processing millions of records becomes not just feasible but economical. On an 8×A100 setup with data parallelism, it is realistic to sustain **200,000+ tokens per second** across the full machine — enough to process 10 million records in hours rather than days.

The keys to success are:

1.  **Understanding the Memory Bottleneck:** KV cache management is the primary constraint. vLLM's PagedAttention addresses this at the engine level, but you still need to configure memory parameters thoughtfully.
2.  **Choosing the Right Parallelism Strategy:** Use data parallelism for models that fit on a single GPU to maximize throughput. Consider quantization to keep larger models within the data parallelism regime. Fall back to tensor parallelism only for models that truly cannot fit on a single GPU.
3.  **Tuning Engine Parameters for Your Workload:** Carefully configure `gpu_memory_utilization`, `max_num_batched_tokens`, and `max_num_seqs` to maximize hardware utilization without running out of memory. Remember that optimal settings depend on your specific input/output length distribution.
4.  **Building a Robust Pipeline:** Implement checkpointing, incremental result writing, and real-time monitoring. The inference engine is fast — make sure the rest of your pipeline can keep up.
5.  **Measuring Everything:** Implement robust logging to calculate Tokens per Second, enabling you to accurately benchmark and iteratively optimize your end-to-end pipeline.

By following these principles and using the code patterns illustrated in this guide, you can build a production-grade system for all your batch LLM inference needs.

---

### References

[1] Zhou, Z. (2024). *vllm_batch_inference*. GitHub. [https://github.com/zxzhou/vllm_batch_inference](https://github.com/zxzhou/vllm_batch_inference)

[2] Kwon, W., Li, Z., Zhuang, S., et al. (2023). *Efficient Memory Management for Large Language Model Serving with PagedAttention*. Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles. [https://github.com/vllm-project/vllm](https://github.com/vllm-project/vllm)

[3] Lin, J., Tang, J., Tang, H., et al. (2024). *AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration*. MLSys 2024. [https://github.com/mit-han-lab/llm-awq](https://github.com/mit-han-lab/llm-awq)
