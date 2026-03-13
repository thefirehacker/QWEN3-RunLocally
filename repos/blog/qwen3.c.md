---
title: "Running Qwen3 0.6B Locally in Pure C — A Deep Dive"
description: "How a single C file loads a 3 GB model, tokenizes your text, and generates responses — explained from first principles with the actual source code."
date: 2026-03-11
categories: [llm, c, inference, qwen3, transformer]
---

# Running Qwen3 0.6B Locally in Pure C

I ran a large language model on a Mac with a single C binary, no Python, no cloud API, no Ollama. This is the story of how that works — and more importantly, why it works — traced directly through the source code of [qwen3.c](https://github.com/thefirehacker/qwen3.c).

Here is what the running model looks like:

```
Multi-turn = off, thinKing = off, tps(R) = off, ttFt = off, Temperature = 0.60, top-P = 0.95
Press Enter to exit the chat
Enter system prompt (or Enter to skip): what is the best way to run a c file on mac
Q: just answer question regarding c file
A: To run a C file on a Mac, you can use a compiler that supports C, such as GCC or Clang.
   Here are some steps you can follow:

   1. Install a C compiler...
   2. Compile the C file: gcc filename.c
   3. Run the compiled file: ./compiledfile
```

A 3 GB FP32 model, running inference entirely in C, on your laptop. Let us look at every step from start to finish.

---

## Overall Architecture

Before diving in, here is the complete flow of what happens when you type a message:

```{mermaid}
flowchart TD
    A["User types message"] --> B["Chat Template\n(ChatML format)"]
    B --> C["render_prompt string\ne.g. <|im_start|>user..."]
    C --> D["encode()\nBPE Tokenization"]
    D --> E["prompt_tokens[]\ninteger IDs"]
    E --> F["forward() loop\nprefill phase"]
    F --> G["Transformer\n28 layers"]
    G --> H["logits[]\n151936 floats"]
    H --> I["sample()\ntop-p + temperature"]
    I --> J["next token ID"]
    J --> K["decode_token_id()\nback to text"]
    K --> L["print char"]
    L --> M{EOS token?}
    M -- No --> F
    M -- Yes --> A
```

---

## 1. The Model: A Transformer in C Structs

Before any text is processed, the model must be loaded. qwen3.c represents the entire Qwen3 architecture as plain C structs.

### 1.1 Configuration

```c
typedef struct {
    int dim;        // embedding dimension (1024 for 0.6B)
    int hidden_dim; // FFN hidden size (3072)
    int n_layers;   // number of transformer blocks (28)
    int n_heads;    // query attention heads (16)
    int n_kv_heads; // key/value heads (8) — Grouped Query Attention
    int vocab_size; // number of tokens (151936)
    int seq_len;    // max context length (32768)
    int head_dim;   // per-head dimension (128)
} Config;
```

These are the hyperparameters of the Qwen3 0.6B model. `n_kv_heads < n_heads` means the model uses **Grouped Query Attention (GQA)** — a memory-saving trick where 2 query heads share one key/value head.

### 1.2 Weights

```c
typedef struct {
    float* token_embedding_table; // (vocab_size=151936, dim=1024)
    float* wq;  // query projection (layer, dim, n_heads*head_dim)
    float* wk;  // key projection   (layer, dim, n_kv_heads*head_dim)
    float* wv;  // value projection (layer, dim, n_kv_heads*head_dim)
    float* wo;  // output projection
    float* w1;  // FFN up
    float* w2;  // FFN down
    float* w3;  // FFN gate
    float* rms_att_weight;   // attention layer norm weights
    float* rms_ffn_weight;   // FFN layer norm weights
    float* rms_final_weight; // final layer norm
    float* wcls;             // output projection to vocab
} TransformerWeights;
```

Every weight is a raw `float*` pointer. There are no tensors or frameworks — just arrays of 32-bit floats.

### 1.3 Loading with mmap

The GGUF file is loaded without copying it into RAM — it is memory-mapped:

```c
*data = mmap(NULL, *file_size, PROT_READ, MAP_PRIVATE, *fd, 0);
void* weights_ptr = ((char*)*data) + 5951648; // skip GGUF header
memory_map_weights(weights, config, weights_ptr);
```

The OS streams pages from disk as they are accessed. The weight pointers are just offsets into the file on disk. This is why the model starts immediately even though it is 3 GB.

### Architecture Diagram

```{mermaid}
flowchart LR
    subgraph file ["GGUF File (3 GB on disk)"]
        H["Header\n5.9 MB"] 
        W["Weights\n~2.4 GB FP32"]
    end
    subgraph memory ["Virtual Memory (mmap)"]
        P["float* pointers\nzero-copy"]
    end
    subgraph structs ["C Structs"]
        TW["TransformerWeights\nwq, wk, wv, wo..."]
        RS["RunState\nx, q, k, v, logits..."]
    end
    file --> memory --> structs
```

---

## 2. System Prompt and Chat Templates

### What is a System Prompt?

A **system prompt** is an instruction you give the model before the conversation starts. It sets the model's role, tone, or persona. In the demo above, the system prompt was: *"what is the best way to run a c file on mac"* — which told the model to focus its answers on that context.

### What is a Chat Template?

An LLM does not understand the concepts of "user" and "assistant" natively. It only sees a flat sequence of tokens. A **chat template** is a formatting convention that marks who is speaking using special tokens.

Qwen3 uses the **ChatML** format. Here is how qwen3.c builds the prompt string (line 962–968):

```c
// With system prompt:
char system_template[] =
    "<|im_start|>system\n%s<|im_end|>\n"
    "<|im_start|>user\n%s<|im_end|>\n"
    "<|im_start|>assistant\n";

// Without system prompt:
char user_template[] =
    "<|im_start|>user\n%s<|im_end|>\n"
    "<|im_start|>assistant\n";
```

`<|im_start|>` and `<|im_end|>` are special tokens in the vocabulary. They tell the model "this is where a turn begins/ends". The model learned during training to generate a reply after seeing `<|im_start|>assistant\n`.

### Suppressing Chain-of-Thought

Qwen3 supports a reasoning (thinking) mode that emits `<think>...</think>` blocks. When thinking is **off** (the default), the code injects an empty think block (line 971):

```c
if (!think_on) {
    strcat(rendered_prompt, "<think>\n\n</think>\n");
}
```

This tricks the model into skipping the reasoning phase and going straight to the answer.

### Full Template Flow

```{mermaid}
flowchart TD
    SP["system_prompt\n(optional)"]
    UP["user_prompt\n(from stdin)"]
    SP --> T["sprintf into rendered_prompt"]
    UP --> T
    T --> RP["rendered_prompt:\n<|im_start|>system\n...<|im_end|>\n<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n"]
    RP --> E["encode()"]
```

---

## 3. Tokenization — From Text to Numbers

The transformer cannot accept text directly. Everything must become integers. The process of converting text to integers is called **tokenization**.

### Vocabulary

The Qwen3 0.6B model has a vocabulary of **151,936 tokens**. Each token is a short string — often a word fragment, a punctuation mark, or a special tag like `<|im_start|>`.

```c
char *vocab[MAX_VOCAB]; // 151936 entries loaded from vocab.txt
```

The vocabulary is loaded line-by-line from `vocab.txt` (which was extracted from the GGUF header). Every token has an integer ID equal to its position in this array.

### Special Tokens

Some tokens have structural meaning and are handled before regular tokenization:

```c
const char *special_tokens[] = {
    "<|im_start|>",
    "<|im_end|>",
    "<think>",
    "</think>"
};
```

When the encoder sees one of these strings, it directly returns its vocabulary ID without going through the byte-level split.

### Byte-to-Unicode Mapping

Before BPE merging, each raw byte of input is mapped to a Unicode character. This is a GPT-2 convention: it ensures every possible byte value has a printable representation and a valid token in the vocab. Bytes in the printable ASCII range map directly; others map to codepoints starting at 256.

```c
void init_byte_unicode_map() {
    // printable ASCII (33–126) and Latin-1 ranges map directly
    // other bytes map to 256+n (multi-byte UTF-8 representation)
}
```

### Tokenization Flow

```{mermaid}
flowchart TD
    T["rendered_prompt string"] --> S{Special token?}
    S -- Yes --> ST["Emit special token ID\ne.g. <|im_start|> → 151644"]
    S -- No --> B["Convert each byte\nto unicode representation"]
    B --> BPE["BPE merge loop\nfind best pair rank, merge, repeat"]
    BPE --> MAP["Map merged token string\nto vocab ID"]
    MAP --> IDS["prompt_tokens[]\n[151644, 9639, 198, ...]"]
```

---

## 4. Encoding — BPE in C

**Byte Pair Encoding (BPE)** is the algorithm that builds tokens out of characters. The idea:

1. Start with every character as its own token.
2. Find the pair of adjacent tokens that appears most frequently in training data (the "merge rule").
3. Merge them into one token.
4. Repeat.

The result is a vocabulary where common words are single tokens, and rare words are split into sub-word pieces.

### The encode() Function

The full encoding pipeline (lines 646–728):

**Phase 1 — Character split:**

```c
while (*p) {
    int match_len = 0;
    int special_id = match_special_token(p, &match_len);
    if (special_id >= 0) {
        tokens[count++] = strdup(vocab[special_id]);
        p += match_len;
        continue;
    }
    // convert raw byte to unicode string
    unsigned char b = *p++;
    tokens[count++] = strdup(unicode_bytes[b]);
}
```

**Phase 2 — BPE merge:**

```c
bool changed = true;
while (changed) {
    int best_rank = INT_MAX;
    int best_pos = -1;
    // find the lowest-ranked (highest priority) pair
    for (int i = 0; i < count - 1; i++) {
        int rank = get_merge_rank(tokens[i], tokens[i + 1]);
        if (rank < best_rank) {
            best_rank = rank;
            best_pos = i;
        }
    }
    if (best_pos == -1) break;
    // merge that pair in-place
    char *merged = malloc(MAX_TOKEN_LEN * 2);
    snprintf(merged, MAX_TOKEN_LEN * 2, "%s%s", tokens[best_pos], tokens[best_pos+1]);
    tokens[best_pos] = merged;
    // shift remaining tokens left
    count--;
}
```

**Phase 3 — Map to IDs:**

```c
for (int i = 0; i < count; i++) {
    // linear scan of vocab to find matching string → return index
    for (int j = 0; j < 151936; j++) {
        if (strcmp(tokens[i], vocab[j]) == 0) { id = j; break; }
    }
    token_ids[token_id_count++] = id;
}
```

### Example

Input: `"Hello"` → bytes → unicode strings → `["H","e","l","l","o"]` → BPE merges → `["Hello"]` or `["Hell","o"]` depending on merge rules → IDs like `[9946, 78]`.

---

## 5. The Transformer Forward Pass

This is the core computation. For each token, `forward()` passes it through 28 transformer layers to produce **logits** — a score for every possible next token.

### Token Embedding

The first step: look up the token's vector representation.

```c
memcpy(s->x, w->token_embedding_table + token * p->dim, p->dim * sizeof(float));
```

Token ID `9946` → copy 1024 floats starting at position `9946 * 1024` in the embedding table. This `s->x` vector (1024 floats) is the "current state" that flows through all layers.

### Per-Layer Processing

For each of the 28 layers:

```{mermaid}
flowchart TD
    X["s→x\n(1024 floats)"] --> RN1["RMSNorm\n(pre-attention)"]
    RN1 --> QKV["matmul → Q, K, V\n(wq, wk, wv)"]
    QKV --> QKN["QK RMSNorm\n(per head)"]
    QKN --> RoPE["RoPE\nrotary position encoding"]
    RoPE --> KVC["Write K,V\nto KV Cache"]
    RoPE --> ATT["Multi-head Attention\nQ·K / √d → softmax → ·V"]
    KVC --> ATT
    ATT --> WO["matmul → wo\nproject back to dim"]
    WO --> RES1["Residual\nx += xb2"]
    RES1 --> RN2["RMSNorm\n(pre-FFN)"]
    RN2 --> FFN["FFN: SwiGLU\nw1=up, w3=gate, w2=down"]
    FFN --> RES2["Residual\nx += xb"]
    RES2 --> X2["s→x\n(updated)"]
    X2 --> NL["Next Layer"]
```

### Attention (GQA)

Qwen3 0.6B has 16 query heads but only 8 KV heads. Two Q heads share one K/V pair (`kv_mul = 2`).

```c
float* k = s->key_cache + loff + t * kv_dim + (h / kv_mul) * p->head_dim;
```

For each query head `h`, attention is:

```
score = Q[h] · K[t] / sqrt(head_dim)    for each past position t
att   = softmax(scores)
out   = sum_t(att[t] * V[t])
```

### RoPE — Positional Encoding

Unlike adding a positional vector, **RoPE** rotates the Q and K vectors based on position. This lets the model generalise to longer sequences than it was trained on.

```c
float freq = 1.0f / powf(1000000.0f, (float)i / (p->head_dim/2));
float fcr = cosf(pos * freq);
float fci = sinf(pos * freq);
q[i]               = x_q * fcr - y_q * fci;
q[i + head_dim/2]  = x_q * fci + y_q * fcr;
```

### FFN — SwiGLU

Each layer has a feed-forward network that expands and contracts the representation:

```c
// up-project and gate simultaneously
matmul(s->hb,  s->xb, w->w1 + l*offset, dim, hidden_dim); // up (3072)
matmul(s->hb2, s->xb, w->w3 + l*offset, dim, hidden_dim); // gate

// SwiGLU: gate * silu(up)
for (int i = 0; i < hidden_dim; i++) {
    float val = s->hb2[i];
    val *= (1.0f / (1.0f + expf(-val))); // silu activation
    val *= s->hb[i];
    s->hb2[i] = val;
}

matmul(s->xb, s->hb2, w->w2 + l*offset, hidden_dim, dim); // down
```

### Final Output

After all 28 layers, a final RMSNorm and a matmul with `wcls` produces the logits:

```c
rmsnorm(s->x, s->x, w->rms_final_weight, p->dim);
matmul(s->logits, s->x, w->wcls, p->dim, p->vocab_size);
// s->logits is now 151936 floats — one score per possible next token
```

---

## 6. Sampling — Picking the Next Token

The transformer gives us 151,936 raw scores (**logits**). We need to pick one token. Three parameters control how:

### Temperature

**Temperature** divides the logits before converting them to probabilities. Lower temperature = more confident, more predictable. Higher temperature = more random, more creative.

```c
// inside sample():
for (int q = 0; q < n; q++) {
    logits[q] /= temperature;  // scale logits
}
softmax(logits, n);             // convert to probabilities
```

- **Temperature = 0.0** → always pick the highest-scoring token (greedy, fully deterministic).
- **Temperature = 0.6** (qwen3.c default) → slight randomness, coherent but varied.
- **Temperature = 1.0** → raw model probabilities, more creative but can drift.
- **Temperature > 1.0** → very random, often incoherent.

```{mermaid}
flowchart LR
    L["logits\n[2.1, 0.3, -1.4, ...]"] --> D["÷ temperature (0.6)"]
    D --> SF["softmax\n→ probabilities\n[0.87, 0.09, 0.03, ...]"]
    SF --> TP["top-p filter\n(keep top 95% mass)"]
    TP --> S["sample\nfrom filtered distribution"]
    S --> ID["next token ID"]
```

### Top-p (Nucleus) Sampling

Even after temperature, the distribution may have a long tail of low-probability tokens. **Top-p** sampling keeps only the smallest set of tokens whose cumulative probability exceeds `p` (default 0.95), then samples from only those.

```c
int sample_topp(float* probabilities, int n, float topp, ...) {
    // sort by probability descending
    // find cutoff where cumulative prob >= topp
    // sample only from the top candidates
}
```

### Sampling Flow

```{mermaid}
flowchart TD
    LG["logits[151936]"] --> T["÷ temperature"]
    T --> SM["softmax → probs"]
    SM --> TP{top-p enabled?}
    TP -- Yes --> NS["nucleus: keep top-p% mass\nsort + cumsum cutoff"]
    TP -- No --> SA["sample_mult\nweighted random"]
    NS --> SA
    SA --> NXT["next token ID\ne.g. 9946"]
```

---

## 7. The Chat Loop — Putting It All Together

The `chat()` function (lines 927–1037) is the main loop that drives the whole pipeline.

### State Machine

```{mermaid}
stateDiagram-v2
    [*] --> UserTurn
    UserTurn --> ReadSystemPrompt : pos == 0
    ReadSystemPrompt --> ReadUserPrompt
    UserTurn --> ReadUserPrompt : pos > 0
    ReadUserPrompt --> BlankInput : user hits Enter
    BlankInput --> [*] : exit
    ReadUserPrompt --> RenderTemplate : has input
    RenderTemplate --> Encode : build ChatML string
    Encode --> PrefillPhase : prompt_tokens[]
    PrefillPhase --> GeneratePhase : after last prompt token
    GeneratePhase --> DecodeAndPrint : next token
    DecodeAndPrint --> EOSCheck
    EOSCheck --> GeneratePhase : not EOS
    EOSCheck --> UserTurn : EOS (151645)
```

### Prefill vs. Generation

The loop runs `forward()` at every position `pos`. The behaviour changes based on where you are:

```c
// Prefill: feed prompt tokens one by one
if (pos < num_prompt_tokens) {
    token = prompt_tokens[pos];
}
// Generation: feed the model's own last output
else {
    token = next;
}

float* logits = forward(transformer, token, pos);
next = sample(sampler, logits);
pos++;
```

During **prefill**, the model processes your prompt. During **generation**, it feeds its own output back as input — this is autoregressive generation.

### EOS Detection

When the model generates token ID `151645` (Qwen3's end-of-sequence token), the assistant turn ends:

```c
if (next == 151645) {
    printf("\n");
    user_turn = 1; // back to user
}
```

### Multi-turn and Prefix Caching

With `-m 1` (multi-turn on), all tokens across turns are accumulated in `TokenBuffer tb`:

```c
append_tokens(tb, prompt_tokens, num_prompt_tokens);
```

On the next turn, the model re-runs `forward()` over the entire history. This is **prefix caching** — it ensures the model has full context of the conversation. The cost: TTFT (time to first token) grows with conversation length, but the implementation batches these efficiently.

### Token Decoding

Each generated token ID is decoded back to a string immediately and printed:

```c
char *decoded = decode_token_id(next);
printf("%s", decoded);
fflush(stdout);  // stream output character by character
free(decoded);
```

`fflush(stdout)` is what makes the output appear word-by-word in real time rather than all at once.

---

## 8. Complete Data Flow

Putting every component together:

```{mermaid}
flowchart TD
    subgraph startup ["Startup (once)"]
        GF["GGUF file\nmmap()"] --> TW["TransformerWeights\n(pointers into file)"]
        HT["header.txt"] --> CFG["Config\ndim=1024 layers=28..."]
        VT["vocab.txt"] --> VOC["vocab[151936]"]
        MT["merges.txt"] --> MRG["merges[151386]"]
    end

    subgraph chatloop ["Chat Loop (each turn)"]
        UI["User input"] --> CT["Chat Template\nChatML format"]
        CT --> ENC["encode()\nBPE → token IDs"]
        ENC --> PF["Prefill\nforward() × prompt_len"]
        PF --> GEN["Generate\nforward() × output_len"]
        GEN --> SAM["sample()\ntemp=0.6 top-p=0.95"]
        SAM --> DEC["decode_token_id()"]
        DEC --> OUT["print + fflush"]
        OUT --> EOS{EOS?}
        EOS -- No --> GEN
        EOS -- Yes --> UI
    end

    startup --> chatloop
```

---

## 9. Run It Yourself

```bash
# Clone the repo with submodules
git clone --recurse-submodules https://github.com/thefirehacker/Qwen3-RunLocally
cd Qwen3-RunLocally/repos/qwen3.c

# Download the FP32 model (~3 GB)
git clone https://huggingface.co/huggit0000/Qwen3-0.6B-GGUF-FP32
git lfs pull   # wait for the 3 GB download
mv Qwen3-0.6B-GGUF-FP32/Qwen3-0.6B-FP32.gguf ./

# Build (needs Xcode Command Line Tools on Mac)
make run

# Basic chat
./run Qwen3-0.6B-FP32.gguf

# Faster with OpenMP (replace 8 with your core count)
make runomp
OMP_NUM_THREADS=8 ./run Qwen3-0.6B-FP32.gguf

# Multi-turn + reasoning mode
./run Qwen3-0.6B-FP32.gguf -m 1 -k 1
```

---

## 10. Learning Plan

Use this as a structured way to read the code after this post:

| Phase | Focus | File / Lines | Goal |
|-------|-------|--------------|------|
| 1 | Model loading | `run.c` 1–200 | Understand GGUF mmap, Config, TransformerWeights |
| 2 | Neural net ops | `run.c` 244–434 | Trace rmsnorm, matmul, forward() layer by layer |
| 3 | Tokenization | `run.c` 436–728 | Follow BPE merge loop with a short example string |
| 4 | Sampling | `run.c` 763–902 | Change temperature and observe output differences |
| 5 | Chat loop | `run.c` 926–1037 | Add a printf to see token IDs as they are generated |

### Suggested Exercises

- **Temperature experiment:** try `-t 0.1` (deterministic) vs `-t 1.5` (chaotic). What changes?
- **Print tokens:** add `printf("[%d]", token);` in the generation section of `chat()` to see the raw IDs.
- **Trace BPE:** add `printf` inside `encode()` to watch tokens merge step by step.
- **KV cache size:** calculate how much RAM the KV cache uses: `2 × n_layers × seq_len × n_kv_heads × head_dim × 4 bytes`.

---

## Summary

qwen3.c is remarkable in its simplicity. A single 1130-line C file does everything:

- Memory-maps a 3 GB model with zero copying
- Implements BPE tokenization from scratch
- Runs a full 28-layer transformer with GQA, RoPE, and SwiGLU in under 300 lines
- Samples with temperature and nucleus sampling
- Manages a multi-turn chat loop with prefix caching

Every piece of modern LLM inference, reduced to standard C. No dependencies. No framework. Just floats and pointers.
