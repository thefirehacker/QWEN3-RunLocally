---
title: "Qwen3ForCausalLM vs Pure C: Where the Same Model Lives Without a Class"
description: "How Hugging Face Transformers organizes Qwen3 as Qwen3ForCausalLM—and how the same causal LM inference is expressed in qwen3.c with structs, mmap, and one forward() loop."
date: 2026-03-16
categories: [transformers, qwen3, inference, c, llm]
---

# Qwen3ForCausalLM vs Pure C: Same Pipeline, Different Packaging

If you open [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) on Hugging Face, you see `config.json`: `hidden_size`, `num_hidden_layers`, `num_attention_heads`, `num_key_value_heads`, `intermediate_size`, `rope_theta`, and `architectures: ["Qwen3ForCausalLM"]`. That JSON is the **blueprint** PyTorch uses to build a Python object graph.

This post asks: **where does that blueprint go when there is no `Qwen3ForCausalLM` class—only a single C file?**

We compare:

- **Transformers:** `Qwen3ForCausalLM` — modules, `forward`, `generate`, tokenizer wrapper.
- **qwen3.c:** [`run.c`](https://github.com/thefirehacker/qwen3.c/blob/main/run.c) — `Config`, `TransformerWeights`, `RunState`, `forward()`, `chat()`.

Same causal LM math for inference; totally different organization.

---

## What `Qwen3ForCausalLM` Actually Bundles

In Hugging Face Transformers, `Qwen3ForCausalLM` is a **composition**:

| Piece (conceptual) | Role |
|-------------------|------|
| **Config** (`Qwen3Config`) | Holds hyperparameters from `config.json`. |
| **Model body** (`Qwen3Model`) | Embeddings + stack of decoder layers (attention + MLP per layer). |
| **LM head** | Projects hidden states to vocabulary logits (often tied to embeddings). |
| **Tokenizer** | Text ↔ token IDs (Rust/Python; separate from the model class). |

Training/inference APIs (`forward`, `generate`) orchestrate tensor shapes, attention masks, KV caching inside the layer stack, and sampling **outside** the core forward in `generate`.

You never allocate raw floats by hand—the framework owns tensors, devices, and autograd hooks.

---

## What qwen3.c Bundles Instead

There is **no class**. The program splits responsibilities into explicit structs and functions:

```{mermaid}
flowchart TD
    subgraph tr ["Transformer aggregate"]
        CFG["Config\n(dim, n_layers, ...)"]
        W["TransformerWeights\n(float* to GGUF)"]
        RS["RunState\nactivations + KV cache"]
        FD["mmap pointer + fd"]
    end
    subgraph ops ["Functions"]
        LC["load_config()"]
        RC["read_checkpoint() / mmap"]
        MW["memory_map_weights()"]
        FWD["forward()"]
        CH["chat()"]
    end
    CFG --> LC
    W --> MW
    RS --> FWD
    tr --> FWD
    tr --> CH
```

Rough analog:

| Transformers idea | qwen3.c location |
|-------------------|------------------|
| `Qwen3Config` | `Config` struct + `load_config()` reading `header.txt` |
| Parameter tensors (`nn.Linear`, embeddings, norms) | `TransformerWeights` — `float*` slices into mmap’d GGUF |
| Hidden states / kv cache | `RunState` — `calloc` buffers + `key_cache` / `value_cache` |
| `model.forward(...)` | `forward(Transformer*, token, pos)` |
| `model.generate(...)` | `chat(...)` — token loop, sampling, printing |
| Tokenizer | `Tokenizer` + `encode()` / `decode_token_id()` + `vocab.txt` / `merges.txt` |

---

## Config: `config.json` vs `Config` + `header.txt`

Official Hugging Face models ship **`config.json`**. For Qwen3 0.6B you typically see:

- `hidden_size` → embedding dimension  
- `num_hidden_layers` → depth  
- `num_attention_heads` / `num_key_value_heads` → GQA layout  
- `intermediate_size` → FFN inner width  
- `max_position_embeddings` → context budget  
- `rope_theta` → RoPE base (used in attention code)

In qwen3.c, **the same numbers** are loaded from a **text dump** of GGUF metadata called `header.txt`, keyed by names such as `QWEN3_EMBEDDING_LENGTH`, `QWEN3_BLOCK_COUNT`, etc.:

```193:241:repos/qwen3.c/run.c
void load_config(Transformer *t) {
    FILE *f = fopen("header.txt", "r");
    // ...
            if (strcmp(key, "QWEN3_EMBEDDING_LENGTH") == 0) {
                t->config.dim = atoi(val);
            } else if (strcmp(key, "QWEN3_FEED_FORWARD_LENGTH") == 0) {
                t->config.hidden_dim = atoi(val);
            } else if (strcmp(key, "QWEN3_BLOCK_COUNT") == 0) {
                t->config.n_layers = atoi(val);
            } else if (strcmp(key, "QWEN3_ATTENTION_HEAD_COUNT") == 0) {
                t->config.n_heads = atoi(val);
            } else if (strcmp(key, "QWEN3_ATTENTION_HEAD_COUNT_KV") == 0) {
                t->config.n_kv_heads = atoi(val);
            } else if (strcmp(key, "QWEN3_CONTEXT_LENGTH") == 0) {
                t->config.seq_len = atoi(val);
            } else if (strcmp(key, "QWEN3_ATTENTION_KEY_LENGTH") == 0) {
                t->config.head_dim = atoi(val);
            }
```

The **`Config` struct** is the C-side mirror of the JSON:

```18:27:repos/qwen3.c/run.c
typedef struct {
    int dim; // transformer dimension
    int hidden_dim; // for ffn layers
    int n_layers; // number of layers
    int n_heads; // number of query heads
    int n_kv_heads; // number of key/value heads (can be < query heads because of multiquery)
    int vocab_size; // vocabulary size
    int seq_len; // max sequence length
    int head_dim; // attention dimension
} Config;
```

**Takeaway:** Transformers reads JSON into Python objects; qwen3.c reads key/value lines into integers. Same hyperparameters, different file format.

---

## Weights: `nn.Module` Parameters vs `TransformerWeights`

In PyTorch, each layer owns parameters (`weight`, `bias`) registered on submodules. In qwen3.c, **all weights are one contiguous FP32 region** after skipping the GGUF header, reinterpreted as `float*` with manual pointer arithmetic:

```121:154:repos/qwen3.c/run.c
void memory_map_weights(TransformerWeights* w, Config* p, void* pt) {
    float *ptr = (float*) pt; 

    w->wcls = ptr;
    ptr += p->vocab_size * p->dim;
    w->rms_final_weight = ptr;
    ptr += p->dim;
    w->token_embedding_table = ptr;
    ptr += p->vocab_size * p->dim;
    w->wk = ptr;
    // ... advances ptr through wk_norm, rms_att_weight, wo, wq, wq_norm, wv,
    // w2, w3, rms_ffn_weight, w1 — tensor order matches GGUF packing
}
```

Loading uses **`mmap`**—no full-file `malloc`, OS-backed pages:

```158:176:repos/qwen3.c/run.c
void read_checkpoint(char *checkpoint, Config *config, TransformerWeights* weights, int* fd, float** data, ssize_t* file_size) {
    // ...
    *data = mmap(NULL, *file_size, PROT_READ, MAP_PRIVATE, *fd, 0);
    void* weights_ptr = ((char*)*data) + 5951648; // skip GGUF header bytes
    memory_map_weights(weights, config, weights_ptr);
}
```

**Takeaway:** Transformers loads shards (`safetensors`, FP16/BF16, device placement). qwen3.c assumes a **single FP32 GGUF** layout and maps tensors by offset. Same tensors, different loader.

---

## Forward Pass: `Qwen3ForCausalLM.forward` vs `forward()`

Transformers’ forward returns logits (and optionally past_key_values, attentions, etc.) from batched tensors.

qwen3.c exposes **one token step at a time**: current token ID + position index → logits vector:

```297:297:repos/qwen3.c/run.c
float* forward(Transformer* transformer, int token, int pos) {
```

Inside: embedding lookup → loop over `n_layers` with attention (including RoPE and KV cache write/read) → SwiGLU FFN → residual paths → final RMSNorm → matrix multiply into `logits` (`wcls`).

**Takeaway:** The **sequence of operations** matches a decoder-only transformer forward pass; the API shape is minimal (scalar token in, logits out) because the chat loop owns the sequence.

---

## Generation: `generate()` vs `chat()`

Transformers `generate()` wraps:

- prompt encoding  
- prefill  
- autoregressive decoding  
- stopping criteria  
- repetition penalty, etc. (optional)

qwen3.c **`chat()`** hardcodes the interactive UX: read system/user prompts, apply ChatML templates, `encode()`, then alternate **prefill** (consume prompt tokens) and **decode** (sample next token, stream print) until EOS.

Sampling (`temperature`, top-p) lives in **`sample()`** / `Sampler`, analogous to Hugging Face’s logits processors + sampler—implemented manually in C.

**Takeaway:** `chat()` ≈ a narrow, stdin/stdout version of `generate()` for this demo binary—not the full Generate API surface.

---

## Tokenizer: Separate from the Model Class (Both Worlds)

Transformers keeps **`AutoTokenizer`** separate from `Qwen3ForCausalLM`.

qwen3.c likewise builds lexers from shipped files:

- `vocab.txt`, `merges.txt`  
- `encode()` / `decode_token_id()`  

So both stacks agree: **tokenization is not “inside” the causal LM class**—it is a paired component.

---

## What Pure C Does *Not* Include

Be precise when comparing:

| Feature | Transformers `Qwen3ForCausalLM` | qwen3.c `run.c` |
|---------|----------------------------------|-----------------|
| Training / gradients | Yes (with trainer) | No |
| `safetensors` / BF16 weights | Typical path | FP32 GGUF path assumed |
| Batched inference | Yes | Effectively single sequence |
| FlashAttention / fused kernels | Often available | Plain loops + optional OpenMP |
| Full `generate()` options | Large surface | CLI flags only |

So the blog theme is **“same causal LM inference core, stripped to inference-only C.”**

---

## Summary Table

| Concept | Hugging Face Transformers | qwen3.c pure C |
|---------|---------------------------|----------------|
| Blueprint | `config.json` → `Qwen3Config` | `header.txt` → `Config` |
| Modules | `Qwen3Model`, layers, norms, heads | No modules; `TransformerWeights` pointers |
| Parameters | `state_dict` / safetensors | mmap GGUF → `memory_map_weights` |
| Forward | `forward(hidden_states, ...)` tensors | `forward(transformer, token, pos)` |
| KV cache | `past_key_values` objects | `key_cache` / `value_cache` arrays |
| Generation | `generate()` | `chat()` + `sample()` |
| Tokenizer | `AutoTokenizer` | `encode` / `decode` + text vocab files |

---

## Why This Comparison Helps Learners

`Qwen3ForCausalLM` hides complexity behind good defaults. **`run.c` forces you to see the buckets:** config, weights, activations, cache, one forward step. Reading both:

1. Connects **JSON hyperparameters** to **layout and loop bounds** in C.  
2. Shows that **classes are not the model**—they are organizational sugar over tensors and control flow.  
3. Makes **KV cache** and **autoregressive decoding** concrete because they appear as explicit arrays and loops.

If you are following [First Break AI — Step 2](https://thefirehacker.github.io/firstbreakai/roadmap.html) (“run a model locally”), use Transformers for breadth and qwen3.c for depth.

---

## Further Reading

- Companion deep dive (tokens, templates, attention): [`qwen3.c.md`](./qwen3.c.md) in this repo.  
- Upstream inference code: [thefirehacker/qwen3.c](https://github.com/thefirehacker/qwen3.c).  
- Official model card: [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B).
