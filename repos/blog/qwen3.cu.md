---
title: "From qwen3.c to qwen3.cu: The Same Forward Pass on a GPU"
description: "You understand qwen3.c. This post walks through runcu.cu — what moved to the GPU, why each kernel exists, and what this educational port gets right vs wrong compared to production inference."
date: 2026-06-18
categories: [qwen3, cuda, inference, gpu, llm]
---

> **Prerequisite:** [Running Qwen3 0.6B Locally — LLM Inference from First Principles](qwen3.c.md). This post assumes you have read `forward()` in `run.c`, understand attention, the KV cache, and know that matmul dominates runtime.

---

# From qwen3.c to qwen3.cu: The Same Forward Pass on a GPU

In [`run.c`](../qwen3.c/run.c), the comment above `matmul()` says it plainly:

```c
// by far the most amount of time is spent inside this little function
```

That function is a double loop. OpenMP parallelizes the outer loop across CPU cores, but Qwen3 0.6B on a Mac still runs at low single-digit tokens per second. Each generated token calls `forward()` once. Inside `forward()`, every layer runs seven matrix-vector products — three for Q/K/V, one for the attention output projection, three for the FFN. For Qwen3 0.6B that means `matmul(s->q, s->xb, w->wq, n=1024, d=2048)`: 1024 × 2048 = **2,097,152 multiply-adds** for the query projection alone. Times three for Q/K/V. Times 28 layers. Per token.

[`runcu.cu`](../qwen3.cu/runcu.cu) does not change the model. The attention math, the KV cache layout, the ChatML template, the sampler — all identical. What changes is where those multiply-adds happen and how many run concurrently. The tokenizer, chat loop, and `sample()` stay on the CPU. Only `forward()` moves to the GPU.

This post walks through that port: what lives in GPU memory, how `forward()` becomes a sequence of kernel launches, and why each kernel exists — with numbers from the actual Qwen3 0.6B config (`dim=1024`, `n_heads=16`, `n_layers=28`, `head_dim=128`, `hidden_dim=3072`).

---

## What moved to the GPU

In `run.c`, weights are memory-mapped from the GGUF file into CPU RAM. Activations are `calloc`'d on the host. Every operation reads and writes CPU pointers directly.

In `runcu.cu`, the startup sequence is different. Weights are still `mmap`'d on the host first — then copied once to device VRAM:

```c
// runcu.cu — read_checkpoint()
CUDA_CHECK(cudaMalloc((void**)&d_weights_ptr, weights_size));
CUDA_CHECK(cudaMemcpy(d_weights_ptr, (*data) + 5951648/4, weights_size, cudaMemcpyHostToDevice));
memory_map_weights(weights, config, d_weights_ptr);
```

All activation buffers — `x`, `q`, `k`, `v`, the KV cache, attention scores — are allocated on the device in `malloc_run_state`:

```c
CUDA_CHECK(cudaMalloc(&s->x, p->dim * sizeof(float)));
CUDA_CHECK(cudaMalloc(&s->key_cache, p->n_layers * p->seq_len * kv_dim * sizeof(float)));
CUDA_CHECK(cudaMalloc(&s->value_cache, p->n_layers * p->seq_len * kv_dim * sizeof(float)));
// ... q, k, v, att, hb, hb2, xb, xb2, xb3, d_logits
```

`s->logits` alone stays on the host (`calloc`), because `sample()` runs on the CPU.

During generation, only two PCIe transfers happen per token. The embedding row goes in:

```c
CUDA_CHECK(cudaMemcpy(s->x, content_row, p->dim * sizeof(float), cudaMemcpyHostToDevice));
```

That is 1024 floats = 4 KB. After all 28 layers, logits come back:

```c
matmul(s->d_logits, s->x, w->wcls, p->dim, p->vocab_size);
CUDA_CHECK(cudaMemcpy(s->logits, s->d_logits, p->vocab_size * sizeof(float), cudaMemcpyDeviceToHost));
```

That is 151,936 floats ≈ 608 KB. Weights (~2.4 GB) stay on the GPU for the entire session.

| Data | qwen3.c | qwen3.cu |
|------|---------|----------|
| Weights | `mmap` → CPU RAM | `cudaMemcpy` once → VRAM |
| Activations | `calloc` on CPU | `cudaMalloc` on GPU |
| Logits for sampling | computed in place | computed on GPU, copied back |

The reason every op in `forward()` becomes a kernel — not just matmul — is that activations already live on the device. `matmul` expects `s->xb` as a GPU pointer; you cannot call the CPU loop without copying data back and forth each step. Once the embedding row is on the GPU, the entire layer chain stays there.

One simplification worth flagging: copying 608 KB of logits per token to run `sample()` on the CPU is wasteful. Production systems sample on the GPU or copy only the top-k candidates. This port prioritizes readability over that optimization.

---

## `forward()` as a launch schedule

Open `runcu.cu` at `forward()` (line 676) alongside `run.c` (line 297). The structure is the same: embed token, loop 28 layers, final RMSNorm, classifier matmul. The difference is that every function call inside the layer loop now launches a CUDA kernel.

A kernel is a `__global__` function executed on the GPU. The CPU launches it with:

```c
kernel_name<<<gridDim, blockDim, sharedMemBytes>>>(args);
```

This fires `gridDim` blocks of `blockDim` threads each, all running the same function on different data indices. You do not need more syntax than that to read this file.

Here is one transformer layer as an execution graph:

```{mermaid}
flowchart TD
    xin["x dim=1024 on GPU"]

    xin --> rmsA["rmsnorm_kernel"]
    rmsA --> wq["matmul_kernel Wq 1024→2048"]
    rmsA --> wk["matmul_kernel Wk 1024→1024"]
    rmsA --> wv["matmul_kernel Wv 1024→1024"]
    wq --> qknQ["rmsnorm_kernel_multihead ×16"]
    wk --> qknK["rmsnorm_kernel_multihead ×8"]
    qknQ --> ropeQ["RoPe_rotation_kernel ×16"]
    qknK --> ropeK["RoPe_rotation_kernel ×8"]
    ropeQ --> attn["multi_head_attention_kernel ×16"]
    ropeK --> attn
    wv --> attn
    attn --> wo["matmul_kernel Wo 2048→1024"]
    wo --> res1["accum_kernel"]
    xin -->|"residual"| res1

    res1 --> rmsF["rmsnorm_kernel"]
    rmsF --> w1["matmul_kernel w1 1024→3072"]
    rmsF --> w3["matmul_kernel w3 1024→3072"]
    w1 --> silu["f_silu_elementwise_mul_w3_kernel"]
    w3 --> silu
    silu --> w2["matmul_kernel w2 3072→1024"]
    w2 --> res2["accum_kernel"]
    res1 -->|"residual"| res2
```

Per layer, that is roughly twelve kernel launches. Times 28 layers gives ~336 launches per token, plus a final RMSNorm and classifier matmul. Each launch has CPU-side overhead on the order of microseconds. At 336 launches, that overhead alone becomes a meaningful fraction of the ~35 tokens/sec this port achieves. Production inference fuses operations — RMSNorm fused into the following matmul, attention fused into a single FlashAttention kernel — precisely to cut this launch count.

The mechanical mapping from `run.c` to `runcu.cu` is direct. Where `run.c` has:

```c
rmsnorm(s->xb, s->x, w->rms_att_weight + l * layer_offset, p->dim);
matmul(s->q, s->xb, w->wq + l * layer_offset, p->dim, att_head_dim);
```

`runcu.cu` calls functions with the same names, but those functions launch kernels internally. The call site in `forward()` looks almost unchanged — that is the point. The GPU port is a swap of execution engine, not a rewrite of the algorithm.

---

## `matmul_kernel`: the loop you know, parallelized

The CPU version:

```c
// run.c
#pragma omp parallel for private(i)
for (i = 0; i < d; i++) {
    float val = 0.0f;
    for (int j = 0; j < n; j++) {
        val += w[i * n + j] * x[j];
    }
    xout[i] = val;
}
```

The GPU version assigns one output row per thread:

```c
// runcu.cu — matmul_kernel
int i = blockIdx.x * blockDim.x + threadIdx.x;  // which output row am I?

extern __shared__ float shared_x[];

for (int offset = 0; offset < n; offset += blockDim.x) {
    if (offset + tid < n)
        shared_x[tid] = x[offset + tid];
    __syncthreads();

    if (i < d) {
        // dot product of row i of W with chunk of x
        // ... float4 vectorized loads ...
        if (offset == 0) xout[i] = sum;
        else xout[i] += sum;
    }
    __syncthreads();
}
```

For `matmul(s->q, s->xb, w->wq, n=1024, d=2048)`: 2048 threads each compute one row of the output. OpenMP on a 10-core CPU gives you 10 rows at a time. The GPU gives you 2048. That is the speedup source — same dot product, massively more concurrent ones.

The `shared_x` array exists for bandwidth, not algorithm. Without it, each of the 2048 threads would independently read all 1024 elements of `x` from global VRAM — 2048 × 1024 = 2,097,152 reads of `x`. With `shared_x`, each block of 256 threads loads `x` once into fast on-chip shared memory and all 256 threads in that block reuse it. Global memory is ~100× slower than shared memory; this is the most important optimization in this kernel.

The launch configuration for the Q projection:

```c
// d=2048, block_size=256 → grid_size=8
matmul_kernel<<<8, 256, 256 * sizeof(float)>>>(xout, x, w, 1024, 2048);
```

Eight blocks, 256 threads each, 2048 threads total. Each block allocates 256 floats of shared memory for the cached input vector.

The `float4` vectorized loads process four floats per memory transaction instead of one. For `n=1024`, that is 256 vector loads per row instead of 1024 scalar loads.

This kernel is correct and instructive. It is not what you would ship. The Makefile provides an alternative:

```sh
make runcublas   # compiles with -DUSE_CUBLAS
./runcublas Qwen3-0.6B-FP32.gguf
```

That swaps the custom kernel for `cublasSgemv` — NVIDIA's hand-tuned matrix-vector multiply. Same function signature, roughly 2× the tokens/sec. Matmul at scale is a solved problem at the library level; the educational value of `matmul_kernel` is understanding *what* cuBLAS is doing, not replacing it.

---

## `multi_head_attention_kernel`: correct math, naive memory

You already know this from `run.c`. Per head: compute attention scores `Q·Kᵀ / √d` for all timesteps `0..pos`, softmax, then weighted sum of V vectors. `run.c` parallelizes across heads with `#pragma omp parallel for`.

The GPU version maps each head to a block:

```c
// runcu.cu
int h = blockIdx.x;                          // head index: 0..15
float *q = sq + h * head_size;               // this head's query
float *att = satt + h * seq_len;             // this head's score buffer

for (int t = threadIdx.x; t <= pos; t += blockDim.x) {
    float *k = key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
    float score = 0.0f;
    for (int i = 0; i < head_size; i++)
        score += q[i] * k[i];
    att[t] = score / sqrtf(head_size);
}
__syncthreads();

softmax_gpu(att, pos + 1);
__syncthreads();
```

Phase 1 — scores: 1024 threads in the block split the timestep loop. At `pos=100`, each thread handles at most one timestep. At `pos=3000`, each thread handles about three. The dot product over `head_dim=128` elements runs serially inside each thread — that inner loop is short.

`__syncthreads()` is a barrier: every thread in the block must finish before any proceeds. Needed here because all threads must complete their score writes before `softmax_gpu` reads the full `att` buffer.

Phase 2 — softmax: `softmax_gpu` is a `__device__` function (GPU-only, called from within the kernel). It does the same max-subtract, exp, normalize as the CPU `softmax`, but uses a tree reduction in shared memory to find the max and sum across threads.

Phase 3 — weighted sum: the loop order is swapped compared to `run.c`:

```c
// run.c: outer t, inner i
for (int t = 0; t <= pos; t++)
    for (int i = 0; i < head_dim; i++)
        xb3[i] += att[t] * v[i];

// runcu.cu: outer i, inner t — each thread owns output dimension i
for (int i = threadIdx.x; i < head_size; i += blockDim.x) {
    float val = 0.0f;
    for (int t = 0; t <= pos; t++)
        val += att[t] * v[i];
    xb[i] = val;
}
```

Same result. The GPU version assigns output dimensions to threads because that maps cleanly to the SIMT model — each thread writes one element of the output independently.

**A concrete trace at `pos=3`, head 0:** thread 0 computes `att[0]` (score for timestep 0), thread 1 computes `att[1]`, thread 2 computes `att[2]`, thread 3 computes `att[3]`. Threads 4–1023 idle for the score phase. After sync, `softmax_gpu` normalizes all four scores. After another sync, thread 0 computes `xb[0] = Σ att[t] * v[0][t]` across t=0..3, thread 1 computes `xb[1]`, and so on.

The math is correct. The memory access pattern is not optimized. Each head reads the full K cache and V cache from global VRAM for every timestep. At `pos=4096` (the overwritten `seq_len` in `load_config`), per head per layer:

```
K reads: 4096 timesteps × 128 floats × 4 bytes = 2 MB
V reads: 4096 timesteps × 128 floats × 4 bytes = 2 MB
```

Times 16 heads, times 28 layers — per generated token. This is memory-bound, not compute-bound. The GPU spends time waiting for VRAM, not doing arithmetic. This is exactly the problem [FlashAttention](https://arxiv.org/abs/2205.14135) solves: tile Q, K, and V through on-chip SRAM so you never re-read the full cache from global memory. Understanding this kernel means understanding why FlashAttention was worth inventing — the attention equation is unchanged; the memory hierarchy access is the bottleneck.

---

## The other kernels

The remaining operations are shorter to explain because they follow the same patterns already established.

### RMSNorm

RMSNorm needs a sum of squares across the entire vector before it can scale. One thread cannot compute that alone. The kernel uses a two-phase reduction:

1. Each thread sums the squares of its assigned elements into `__shared__ sdata[]`
2. Tree reduction: threads pairwise add (`stride >>= 1`) until one value remains
3. Thread 0 computes `1 / sqrt(mean_sq + ε)` and broadcasts via shared memory
4. All threads apply the scale

For the full hidden state (`dim=1024`): `rmsnorm_kernel<<<1, 1024>>>` — one block, 1024 threads, each handles one element.

For QK-Norm across heads: `rmsnorm_kernel_multihead<<<16, 1024>>>` for Q (16 blocks, one per head), `<<<8, 1024>>>` for K (8 KV heads). `blockIdx.x` selects the head; the reduction runs inside each block independently.

### RoPE

`RoPe_rotation_kernel_multihead<<<n_heads, head_dim/2>>>` — embarrassingly parallel. `blockIdx.x` is the head, `threadIdx.x` is the rotation pair index. Each thread rotates two floats (the `(x, y)` pair at positions `i` and `i + head_dim/2`). For Qwen3: `<<<16, 64>>>` for Q, `<<<8, 64>>>` for K. Production code fuses this into the Q/K projection kernel to avoid a separate launch.

### SwiGLU and residual

The FFN activation in `run.c`:

```c
for (int i = 0; i < hidden_dim; i++) {
    float val = s->hb2[i];
    val *= (1.0f / (1.0f + expf(-val)));  // silu
    val *= s->hb[i];                        // gate multiply
    s->hb2[i] = val;
}
```

becomes `f_silu_elementwise_mul_w3_kernel<<<12, 256>>>` — 3072 threads, one per element of `hidden_dim`. No cooperation needed; each thread is independent.

Residual connections (`s->x[i] += s->xb2[i]`) become `accum_kernel<<<4, 256>>>` for `dim=1024`. Same pattern: one thread per element.

These kernels exist for the same reason as matmul: the vectors are already on the GPU, and 3072 independent element-wise operations are faster with 3072 threads than one CPU loop.

---

## Performance and what this port does not do

The README reports ~35–39 tokens/sec with custom kernels, roughly doubled with `make runcublas`. That is a real improvement over CPU inference, but it is not competitive with production tools. Three concrete reasons:

**1. Launch overhead.** ~336 kernel launches per token. At ~10 µs per launch, that is ~3 ms of pure scheduling overhead before any arithmetic. Fused kernels in llama.cpp and vLLM reduce this to dozens of launches per token by combining RMSNorm + matmul, or the entire attention pass into one kernel.

**2. Naive attention memory access.** As shown in the attention section, this kernel re-reads the full K and V cache from global VRAM every token. FlashAttention and PagedAttention (vLLM) exist specifically to fix this. The attention math is identical; the memory access pattern is the problem.

**3. FP32, no quantization.** This port uses full-precision weights (~2.4 GB). A Q4 GGUF model is 4× smaller and 4× less memory bandwidth per matmul. llama.cpp on the same GPU with Q4 quantization reaches hundreds of tokens/sec — not because the algorithm changed, but because each weight read transfers fewer bytes.

What stayed on the CPU is fine. Tokenization and sampling are negligible compared to the forward pass. What stayed naive — the 608 KB logits copy per token — is worth knowing about but does not dominate.

### What to read next

Each item connects directly to something in this post:

- **[FlashAttention paper](https://arxiv.org/abs/2205.14135)** — read after the attention kernel section. Same Q·Kᵀ softmax Σ(att·V) math; different tiling through SRAM.
- **llama.cpp `ggml-cuda`** — compare their matmul and attention kernels to `matmul_kernel` and `multi_head_attention_kernel` here. Same ops, production-grade memory access.
- **Profile with `nsys`** — on an NVIDIA machine, run `nsys profile ./runcu model.gguf` and confirm matmul and attention dominate, launch overhead is visible.
- **Quantization and vLLM** — the [First Break AI Step 3 roadmap](https://thefirehacker.github.io/firstbreakai/roadmap.html) covers why Q4 GGUF and continuous batching matter for serving.

---

## Summary

| Op | `run.c` | `qwen3.cu` | Production |
|----|---------|------------|------------|
| Matmul | OpenMP loop | `matmul_kernel` / cuBLAS | cuBLAS, CUTLASS, tensor cores |
| Attention | OpenMP per head | `multi_head_attention_kernel` | FlashAttention, paged KV |
| RMSNorm | CPU loop | reduction kernel | fused into matmul |
| RoPE | nested `for h, for i` | `RoPe_rotation_kernel_multihead` | fused into Q/K projection |
| SwiGLU | CPU `for` loop | `f_silu_elementwise_mul_w3_kernel` | fused into FFN |
| Residual | CPU `for` loop | `accum_kernel` | fused |
| Weights | `mmap` CPU | H2D copy once | GPU-native / quantized |
| Sampling | CPU | CPU (logits D2H copy) | GPU sampling |

`qwen3.cu` is an educational port. The README says so: custom kernels for learning, cuBLAS as a faster fallback, kernel optimization on the TODO list. Its value is that you can read every kernel launch in a single file, compare it directly to the `run.c` loop you already understand, and see exactly where production inference would do something different. Start with `forward()` — it is the same story — then read `matmul_kernel` and `multi_head_attention_kernel`. Those two account for most of the runtime and most of the insight.
