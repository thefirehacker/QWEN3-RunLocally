---
title: "Why is qwen3.c slow when llama.cpp is fast? Same GGUF, same prefill and decode"
description: "A First Break AI note from running Qwen3-0.6B-FP32.gguf in qwen3.c and then in llama.cpp on an Apple M1. Same weights and the same two-phase algorithm. The speed difference is kernels and the GPU, not a different transformer."
date: 2026-08-29
categories: [llm, inference, llama.cpp, qwen3, metal, first-break-ai]
---

> **First Break AI — after Step 2**
>
> You already ran Qwen3 0.6B with [`qwen3.c`](https://github.com/thefirehacker/qwen3.c): mmap a GGUF, tokenize, forward pass, sample. This post is what happens when you point **llama.cpp** at the **same file** on a Mac, and why the teaching C binary still feels slow even though it already does prefill and decode.

---

# Why was our C inference slow, and why is llama.cpp fast?

## The question

`qwen3.c` already does the real inference loop. It loads `Qwen3-0.6B-FP32.gguf`, runs embedding, twenty-eight transformer blocks, the language-model head, and sampling. Later builds added batching and prefix caching, so it is not blindly recomputing the whole history on every token. Prefill and decode are there.

And yet, on the same Mac, the same FP32 GGUF, **llama.cpp** answers in a browser at tens of tokens per second on short chat, and hundreds of tokens per second while chewing a long prompt. `qwen3.c` sits around the README number: about **six tokens per second** with OpenMP on four cores.

So the question is not “does C know about KV cache?” It does. The question is:

**If both programs implement prefill and decode on the same weights, why is llama.cpp so much faster? Is that some kernel, or something else?**

The short answer is: **it is kernels, and it is the device those kernels run on.** llama.cpp did not invent a different Qwen3. It runs the same math as Metal GPU shaders (and a much more careful CPU path when something stays on the host). `qwen3.c` runs the same math as ordinary C loops on the CPU. Prefill and decode only decide *how much* work you do. They do not decide *how fast each multiply runs*.

The rest of this post is that answer in order: what we actually ran, the numbers, what prefill and decode are, why C is still slow, what llama.cpp does instead, and how that showed up in the server log.

---

## What we ran

Hardware: **Apple M1**, 16 GB unified memory, eight logical cores (llama.cpp used four threads).

Weights: [`repos/qwen3.c/Qwen3-0.6B-FP32.gguf`](../qwen3.c/Qwen3-0.6B-FP32.gguf) — about **2.8 GB** on disk, full-precision Qwen3-0.6B.

Teaching engine:

```text
./runba Qwen3-0.6B-FP32.gguf
```

or the OpenMP build. That is the First Break binary: one file of C, mmap, your own matmul and attention.

Production-style engine on the same file (Homebrew llama.cpp, `llama-cli` / `llama serve` — same project, two spellings):

```bash
llama-server \
  -m /Users/booimac/AIEDX/Code/AI/Qwen3-RunLocally/repos/qwen3.c/Qwen3-0.6B-FP32.gguf \
  -ngl 99 \
  --jinja \
  --host 127.0.0.1 --port 8080
```

- **`-m`** — the GGUF. Same bytes `qwen3.c` mmap’s.
- **`-ngl 99`** — offload more layers than the model has (28), so **all layers go to Metal**. There is no CUDA on this Mac.
- **`--jinja`** — apply the chat template stored in the GGUF (`<|im_start|>user` …). That changes *prompt shape*, not matmul speed.

The web UI at `http://127.0.0.1:8080` is `llama serve`’s built-in front end. A short question (“what is capital of India”) and a very large paste were both run through that server. The tables below are those two requests plus the published `qwen3.c` figures.

---

## Basic benchmark: every number we actually discussed

These are not a formal `llama-bench` sweep. They are the **exact `slot print_timing` lines** from the M1 `llama-server` log, plus the **qwen3.c README** CPU figure, plus one external tweet for scale. Treat this as a **lab notebook baseline** for this machine and this GGUF.

### Setup (constant for the llama.cpp rows)

| Item | Value |
|---|---|
| Model file | `Qwen3-0.6B-FP32.gguf` (~2.8 GB) |
| Architecture | Qwen3-0.6B: hidden 1024, 28 layers, 16 Q heads, 8 KV heads, head dim 128 |
| Machine | Apple M1, 16 GB unified LPDDR |
| llama.cpp devices at start | BLAS/Accelerate `0 MiB`; **MTL0** 12124 MiB (12123 free *before load*); CPU view 16384 MiB |
| Threads | `n_threads = 4`, `n_threads_batch = 4`, 8 cores visible |
| CPU flags logged | NEON, ARM_FMA, DOTPROD, ACCELERATE, OPENMP, REPACK; Metal `EMBED_LIBRARY` |
| Serving | `n_parallel = 4`, `kv_unified = true`, 4 slots, `n_ctx = 40960` each |
| Prompt cache cap | `--cache-ram` default **8192 MiB** (empty at first long request: `0 prompts, 0.000 MiB`) |
| Chat formatting | `--jinja`, log: `Chat format: peg-native` |
| GPU offload | `-ngl 99` (all 28 layers on Metal) |

### End-to-end timings

| Run | Engine | Device | Prompt tokens (prefill) | Prefill time | Prefill tok/s | Output tokens (decode) | Decode time | Decode tok/s | Total time | Total tokens | Graphs reused | Truncated |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Published teaching baseline | `qwen3.c` + OpenMP | CPU, 4 threads | n/a (README) | n/a | n/a | generation | n/a | **~6** | n/a | n/a | n/a | n/a |
| Short chat: “capital of India” | `llama-server` | Metal | **17** | 825.15 ms | **20.60** | **11** | 829.42 ms | **13.26** | 1654.57 ms | 28 | 10 | 0 |
| Long paste | `llama-server` | Metal | **12,567** | 26,169.36 ms | **480.22** | **287** | 27,042.70 ms | **10.61** | 53,212.06 ms | 12,854 | 285 | 0 |

Prefill tok/s on the 17-token prompt looks “slow” because **startup and launch overhead is divided by 17**. The 12k prompt is the honest prefill number: about **0.48 ktok/s**. Decode dropped from 13.26 to 10.61 tok/s because every new token rereads a **much larger KV cache**.

### Long-prompt chunked prefill (partial prefill)

Default `--batch-size` is **2048**. The long prompt did not fit in one batch. The log printed progress while KV was filled in waves:

| Checkpoint `n_tokens` | Progress | Wall time so far | Instant / running tok/s (log) |
|---:|---:|---:|---:|
| 2,048 | 0.16 | 3.37 s | 607.22 |
| 4,096 | 0.33 | 5.59 s | 733.08 |
| 6,144 | 0.49 | 8.67 s | 708.91 |
| 8,192 | 0.65 | 12.51 s | 654.64 |
| 10,240 | 0.81 | 17.21 s | 595.16 |
| 10,337 | 0.82 | 19.57 s | 528.11 |
| 12,385 | 0.99 | 24.18 s | 512.24 |
| **12,567** (final `prompt eval`) | 1.00 | **26.17 s** | **480.22** |

That series **is** partial / chunked prefill. The short India prompt had a single `prompt eval / 17 tokens` line and **no** `progress = 0.16 … 0.99` lines, because 17 < 2048.

### Decode live ticks (long run)

| `n_decoded` | `tg` (tok/s) |
|---:|---:|
| 100 | 10.16 |
| 133 | 10.35 |
| 171 | 10.78 |
| 200 | 10.58 |
| 230 | 10.50 |
| 263 | 10.54 |
| **287** (final) | **10.61** |

### Memory we derived (not printed as RSS)

| Quantity | Estimate | How |
|---|---|---|
| Weights | **~2.8 GB** | FP32 GGUF size; loaded once |
| KV per token (F16 K/V, GQA) | **~112 KiB** | 28 × 8 × 128 × 2 × 2 bytes |
| Live KV for 12,853 tokens | **~1.4 GB** | 12,853 × ~112 KiB |
| Full reserved context 40,960 tokens, one unified pool | **~4.4 GB** | if the arena were filled |
| This request’s extra (KV + scratch) | **~1.5–2.5 GB** | on top of weights |
| Process during the long run (order of magnitude) | **~5–7 GB** unified | weights + KV + Metal/runtime + OS share |
| Prompt cache at first long request | **0 MiB** | `cache state: 0 prompts` |

### External scale (not our hardware)

| Source | Prefill | Decode / “thinking+output” | Model |
|---|---|---|---|
| Tim Dettmers on GLM 5.3 (tweet, datacenter-class run) | ~**1 ktok/s** | ~**60 tok/s** | 743B-class, not an M1 |
| This M1, long prompt, 0.6B FP32 | **0.48 ktok/s** | **10.6 tok/s** | same *metric names*, different machine |

`llama-bench` is how you turn these one-shot logs into a Dettmers-style **average** (`pp` = prefill, `tg` = generate), for example:

```bash
llama-bench \
  -m /Users/booimac/AIEDX/Code/AI/Qwen3-RunLocally/repos/qwen3.c/Qwen3-0.6B-FP32.gguf \
  -ngl 99 -p 4096 -n 128 -r 5
```

---

## Prefill and decode are not a speed hack

It is easy to hear “we added prefill and decode” and think that *is* the optimization. It is only the **shape** of the work.

**Prefill** (llama.cpp log: `prompt eval time`): the model sees tokens that already exist — the chat template plus your question. Those tokens can be processed in a **batch**. The important output is not only the last logits. It is **K and V for every layer**, written into a cache.

**Decode** (llama.cpp log: `eval time`): the model emits **one new token**. Query is new. Keys and values for the past are **reread from the cache**, not rebuilt from the original text. Then the new token’s K and V are **appended**.

Without a KV cache you would rerun attention over the whole history on every step. That is the disaster the PDF called “postpone KV cache,” and that early `qwen3.c` later fixed. With a cache, decode cost is roughly “one token through the MLP and attention against a growing cache,” not “the entire prompt again.”

So `qwen3.c` with a correct KV cache is already doing the **algorithmically right** thing. It is still slow because the **inner loops** are slow. llama.cpp on Metal is doing the same two phases. The 12k run spent **26 s in prefill** and **27 s in decode**. Prefill was *fast per token* (2.08 ms). Decode was *slow per token* (94 ms) because it is **memory-bound**: each of 287 tokens pulled on ~1.4 GB of KV.

That split — prefill compute-heavy, decode bandwidth-heavy — is exactly what people mean when they quote “1 ktok/s prefill, 60 tok/s output.” You already have those two meters. The tweet is a bigger GPU and a bigger (usually quantized, often MoE) model.

---

## What `qwen3.c` actually is

`qwen3.c` is a descendant of the llama2.c idea: **one C file you can read**, GGUF tensors in memory order, vocab and merges as text, no Python in the hot path.

A matmul looks like “for each output row, dot the input with a weight row.” RMSNorm is a variance, an `rsqrt`, a scale. Attention is reshape heads, RoPE, scores, softmax, weighted V, output projection. That is **correct Qwen3**. It is also the kind of code a compiler turns into **plain CPU instructions**, with OpenMP maybe splitting rows across four cores.

Four cores on an M1, walking **FP32** weights that barely fit in cache, will not saturate the chip. The GPU on the same package is built to pour those multiplies through many lanes at once. `qwen3.c` never asks it to. There is no Metal, no ggml, no tiled GEMM, no FlashAttention. mmap is a **loading** trick, not a **compute** trick.

If an older `./run` still recomputed the full sequence every decode step, it would be slower still. The current story in the qwen3.c README is that batching and prefix caching fixed the *algorithmic* waste. The remaining gap versus llama.cpp is **implementation and hardware**, not “they forgot prefill.”

---

## What llama.cpp actually is

llama.cpp is a **library** (ggml + `llama.h`) plus tools (`llama cli`, `llama serve`). The transformer is described as a **graph of tensor ops**. Backends implement those ops:

- **Metal** on this Mac (`MTL0`)
- CUDA on NVIDIA
- CPU with NEON / Accelerate / AVX elsewhere

`-ngl 99` means: put every transformer layer’s weights and the corresponding ops on the Metal backend. The 2.8 GB file is still the same FP32. The **multiplies** run as **GPU shaders** — tiled, vectorized, written by people whose job is this inner loop.

On top of that, serving adds things that `qwen3.c` either omits or does in a simpler way:

| Piece | Role | Does it make one matmul faster? |
|---|---|---|
| Metal / ggml kernels | The actual GEMM, norm, attention | **Yes. This is the reason.** |
| Flash-style attention (`-fa auto`) | Less traffic than a full score matrix | Yes, especially long context |
| Graph reuse (`graphs reused = 285`) | Replay a compiled Metal graph on decode | Small but real on token-by-token |
| `--jinja` / peg-native | ChatML wrapping | No |
| Slots, LRU, prompt cache | Multi-request KV reuse | No for the *first* 12k prefill (cache was empty) |
| Chunked `--batch-size 2048` | Fits a long prompt on the GPU in waves | Keeps the GPU busy; does not change the math |

The first long request logged `f_keep = -1`, `sim = 0`, `0 prompts` in the prompt cache. So the 480 tok/s prefill was **not** “we skipped work with a cache.” It was **Metal chewing 12,567 tokens**. That is the cleanest proof that the speed is kernels plus GPU.

`graphs reused` on the short chat was 10; on the long chat 285. That tracks **decode steps**, not “we skipped prefill.”

---

## Why decode is only ~2× the C README, but prefill looks like another world

People expect “GPU = 50×.” On **decode**, a 0.6B FP32 model on an M1 is limited by **how fast you can stream weights and KV**, not by how many ALUs you have. The GPU still wins (13 vs 6 tok/s; 10.6 vs 6 on a fat cache), but the win is modest.

**Prefill** batches thousands of tokens. The GPU finally has a wide problem. 480 tok/s versus a CPU walking the same FP32 is the spectacular number. `qwen3.c` never gets that shape of work onto the GPU, so you never see it.

If you quantized the GGUF to Q4 or Q8, llama.cpp would move **fewer bytes** and decode would jump again. `qwen3.c` is written around **FP32** and would not pick that up automatically.

---

## Unified memory: there is no secret second RAM

The startup banner showing Accelerate at `0 MiB` and MTL0 “all free” is **before** `loading model`. Free at `t = 0` does not mean unused.

On M1 there is **one DRAM pool** (unified LPDDR). CPU and GPU share it. “MTL0 12 GB” and “CPU 16 GB” are two **views**, not 28 GB.

**Live KV** during generate is Metal-visible buffers on that pool. **Prompt cache** (`--cache-ram`) is a **second copy** of token IDs plus KV/state after a slot goes idle, so the shared arena can be cleared and a later HTTP POST can restore the conversation without 26 s of prefill. That copy is **also** the same LPDDR. “Host-side” means process heap / cache list, not a different DIMM.

HTTP chat is stateless: every user message is a new POST with the full `messages` array. The desk (slot) may be wiped. The filing cabinet (prompt cache) is how the next POST skips prefill when the prefix matches. First request: cabinet empty. That is why we paid the 26 s once.

---

## Slots, LRU, peg-native (the serving log, briefly)

`n_parallel = 4` is four **desks**. The long job took **slot 3** by **LRU** with `t_last = -1` (never used). `peg-native` is llama.cpp’s parser applying the Jinja chat template — not “peg-negative.” None of that replaces Metal. It is how a server multiplexes chats.

---

## The honest mapping

```text
Same GGUF
Same embedding → 28 blocks → norm → LM head
Same prefill (fill KV) and decode (read KV)

qwen3.c     = school CPU loops, OpenMP
llama.cpp   = ggml Metal kernels on the same package
qwen3.cu    = you write CUDA kernels; this Mac cannot run them
```

You cannot close the gap by “adding prefill” again. You close it by **running the multiplies on the GPU** (llama.cpp / Metal, or `qwen3.cu` on NVIDIA) or by **calling a real GEMM** (Accelerate, ggml) from C.

---

## What to do with this

1. Keep `qwen3.c` to **see** every tensor. Speed is not its job.
2. Use `llama serve` / `llama cli` when you want the **same file** to feel usable on a Mac.
3. Quote **two** numbers, like the tweet: prefill tok/s and decode tok/s. For this lab: **0.48 ktok/s** and **10.6 tok/s** on the long prompt; **13.3 tok/s** decode on the short one.
4. For a clean average, run `llama-bench` on this GGUF with `-ngl 99`.
5. If you want more decode on this M1, change **bytes** (Q8/Q4 GGUF), not the C teaching loops.

The C program was slow **in spite of** prefill and decode because those names describe **scheduling**, not **throughput**. llama.cpp is fast because **ggml’s Metal kernels** run the same Qwen3 on the **GPU** that was sitting idle while `qwen3.c` looped on four CPU threads.
