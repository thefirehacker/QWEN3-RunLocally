---
title: "Step 2: Run a Model Locally — LLM Inference from First Principles"
description: "A complete beginner guide for First Break AI learners. No prior AI knowledge needed. You will run Qwen3 0.6B on your Mac and understand every single step — tokenization, chat templates, attention, sampling, and the KV cache — all traced through the actual source code."
date: 2026-03-11
categories: [llm, inference, tokenization, attention, qwen3, first-break-ai]
---

> **First Break AI — Step 2: Run a Model Locally**
>
> This post is part of the [First Break AI](https://thefirehacker.github.io/firstbreakai/) cohort roadmap.
> Step 2 goal: run a model locally and understand *inference, chat templates, tokenization, and prompting*.
> No AI background needed. Start here.

---

# Running Qwen3 0.6B Locally — LLM Inference from First Principles

By the end of this post you will have:

- Run a real LLM on your Mac with zero cloud APIs
- Understood what a token is and how BPE tokenization works
- Understood what a chat template and system prompt are — and seen the exact template in the C source code
- Understood how the transformer generates text one token at a time
- Understood what attention, temperature, and a KV cache actually are
- A reading and exercise plan to go deeper

Here is the model running on a Mac. We built this with a single C binary:

```
Multi-turn = off, thinKing = off, Temperature = 0.60, top-P = 0.95
Press Enter to exit the chat
Enter system prompt (or Enter to skip): what is the best way to run a c file on mac
Q: just answer question regarding c file
A: To run a C file on a Mac, you can use a compiler that supports C, such as GCC or Clang.
   1. Install a C compiler...
   2. Compile the C file: gcc filename.c
   3. Run the compiled file: ./compiledfile
```

No Python. No Ollama. No cloud. Just a 3 GB model file and a C binary running on your laptop.

Let us understand exactly how this works.

---

## The Big Picture First

Before any detail, here is the complete journey a message takes — from the moment you type it to the moment text appears on screen:

```{mermaid}
flowchart TD
    A["You type a message"] --> B["Chat Template\nWraps it in ChatML format"]
    B --> C["Tokenizer\nBreaks text into tokens"]
    C --> D["Token IDs\nIntegers the model understands"]
    D --> E["Transformer\n28 layers of attention + FFN"]
    E --> F["Logits\n151936 scores, one per token"]
    F --> G["Sampler\nTemperature + top-p picks next token"]
    G --> H["Decoder\nToken ID back to text"]
    H --> I["Printed to screen"]
    I --> J{End of sentence?}
    J -- No --> E
    J -- Yes --> A
```

Each box is a lesson. We will go through all of them, starting from zero.

---

## Lesson 0: What Is an LLM?

Before we write a single command, let us build the right mental model.

### The core idea

A Large Language Model is a program that predicts the next word (actually the next "token") given everything that came before.

That is it. Everything else — the chat, the code it writes, the reasoning — emerges from doing this prediction extremely well, on extremely large amounts of text.

Think of it this way: imagine you read every book, every website, every code repository ever written. Now someone gives you the first half of a sentence and asks "what comes next?" You would have very strong intuitions. An LLM has those intuitions, encoded as billions of numbers.

```{mermaid}
flowchart LR
    P["Everything before:\n'The capital of France is'"] --> M["LLM\n(600M numbers)"]
    M --> N["Next most likely token:\n'Paris'"]
    N --> R["Repeat:\n'The capital of France is Paris'"]
    R --> M2["LLM"]
    M2 --> N2["'.'"]
```

### What makes it "large"?

The "large" in LLM refers to the number of parameters — numbers that were learned during training. Qwen3 0.6B has **600 million parameters**. These are stored as floating-point numbers in the GGUF file we downloaded (3 GB in FP32 format, or about 5 bytes per parameter).

### Why does it feel like chatting?

The model is not "thinking" or "understanding" in the way humans do. It generates text token by token, each time predicting what is most likely to follow. The appearance of understanding comes from the patterns learned across training data, and from the **chat template** — which we will explain next.

---

## Lesson 1: Run It First

We learn best when we can experiment. Let us get it running before explaining further.

### Prerequisites

You need Xcode Command Line Tools (includes `clang` and `make`). If you have not installed them:

```bash
xcode-select --install
```

### Step 1: Get the code

```bash
git clone --recurse-submodules https://github.com/thefirehacker/Qwen3-RunLocally
cd Qwen3-RunLocally/repos/qwen3.c
```

### Step 2: Download the model

The model weights are stored as a GGUF file (~3 GB, FP32 format):

```bash
git clone https://huggingface.co/huggit0000/Qwen3-0.6B-GGUF-FP32
git -C Qwen3-0.6B-GGUF-FP32 lfs pull   # wait — 3 GB download
mv Qwen3-0.6B-GGUF-FP32/Qwen3-0.6B-FP32.gguf ./
```

> **Why is it 3 GB for a 0.6B model?**
> The model has 600 million parameters. Each is stored as a 32-bit float (4 bytes).
> 600,000,000 × 4 bytes = 2.4 GB, plus file headers and metadata ≈ 3 GB total.
> Quantized versions (Q4, Q8) can bring this down to 300–600 MB.

### Step 3: Build and run

```bash
make run
./run Qwen3-0.6B-FP32.gguf
```

You will see:

```
Multi-turn = off, thinKing = off, tps(R) = off, ttFt = off, Temperature = 0.60, top-P = 0.95
Press Enter to exit the chat
Enter system prompt (or Enter to skip):
Q:
```

Type a system prompt (or just press Enter to skip), then type your question. Press Enter with no input to exit.

### Try these now

```
# No system prompt, simple question
Q: What is 2 + 2?

# With system prompt
Enter system prompt: You are a pirate. Respond in pirate speech.
Q: What is the weather like?

# Reasoning mode (adds <think> blocks)
./run Qwen3-0.6B-FP32.gguf -k 1
Q: Why is the sky blue?

# Temperature experiment (compare these)
./run Qwen3-0.6B-FP32.gguf -t 0.1    # very predictable
./run Qwen3-0.6B-FP32.gguf -t 1.5    # more random
Q: Continue this story: Once upon a time...
```

You now have a running LLM. The rest of this post explains what just happened.

---

## Lesson 2: Tokens — The Atoms of Language

The model does not read letters or words directly. It reads **tokens**.

### What is a token?

A token is a chunk of text — usually a word, a word fragment, or a punctuation mark. The model has a fixed vocabulary of **151,936 tokens**. Every possible input must be broken down into these tokens before the model can process it.

```
"Hello, world!" → ["Hello", ",", " world", "!"]
                → [9906,    11,   1917,   0]   ← these are the token IDs
```

### Why not just use letters?

Using individual letters (26 characters) would make sequences very long and expensive to process. Using full words would require an enormous vocabulary and could not handle new words.

**BPE (Byte Pair Encoding)** finds the sweet spot: common words are single tokens, rare words are split into sub-word pieces. "tokenization" might become `["token", "ization"]` — two tokens.

### Why not just use whole words?

- Vocabulary would need millions of entries (one per word in all languages)
- New words, code identifiers, and misspellings would be impossible to handle
- Every language would need a separate tokenizer

BPE handles all of this with a single 151,936-token vocabulary covering English, Chinese, code, and more.

### Tokens in qwen3.c

The vocabulary is loaded from `vocab.txt` (extracted from the GGUF header):

```c
// run.c line 491
void load_vocab(const char *path) {
    // reads vocab.txt line by line
    // vocab[0] = "!" , vocab[1] = "\"", ..., vocab[9906] = "Hello"
    // each token's index IS its token ID
}
```

The vocabulary has 151,936 entries. Every token is just an integer between 0 and 151,935.

### Special tokens

Some tokens are structural markers, not words:

| Token | ID | Meaning |
|-------|-----|---------|
| `<\|im_start\|>` | 151644 | Start of a conversation turn |
| `<\|im_end\|>` | 151645 | End of a conversation turn (also EOS) |
| `<think>` | 151648 | Start of reasoning block |
| `</think>` | 151649 | End of reasoning block |

These are defined in the code:

```c
// run.c line 476
const char *special_tokens[] = {
    "<|im_start|>",
    "<|im_end|>",
    "<think>",
    "</think>"
};
```

### Check your understanding

- What is a token ID?
- Why does Qwen3 have 151,936 tokens and not 26 (letters)?
- What happens if a word is not in the vocabulary?

---

## Lesson 3: Chat Templates and System Prompts

When you type "What is 2 + 2?" to the model, it does not just see that string. The code wraps it in a **chat template** — a structured format that tells the model who is speaking and what role they have.

### Why does a chat template exist?

The raw model only predicts the next token. It has no built-in concept of "user" and "assistant". During training on conversation data, a specific format was used to mark turns. At inference time we must use the same format so the model knows what role it is playing.

Think of it like a film script format. The model learned from millions of scripts that look like:

```
<|im_start|>user
Question or instruction here<|im_end|>
<|im_start|>assistant
Answer here<|im_end|>
```

So at inference time we format our input the same way, and the model picks up naturally — "I'm the assistant, now I generate my reply."

### The ChatML format

Qwen3 uses **ChatML** (Chat Markup Language). Here is exactly what qwen3.c builds (lines 962–968 of `run.c`):

```c
// WITH a system prompt:
char system_template[] =
    "<|im_start|>system\n"
    "%s"                          // your system prompt goes here
    "<|im_end|>\n"
    "<|im_start|>user\n"
    "%s"                          // your question goes here
    "<|im_end|>\n"
    "<|im_start|>assistant\n";   // model generates from here

// WITHOUT a system prompt:
char user_template[] =
    "<|im_start|>user\n"
    "%s"                          // your question goes here
    "<|im_end|>\n"
    "<|im_start|>assistant\n";
```

When you typed "You are a pirate" as the system prompt and "What is the weather?" as your question, the model actually received this token sequence:

```
<|im_start|>system
You are a pirate. Respond in pirate speech.<|im_end|>
<|im_start|>user
What is the weather?<|im_end|>
<|im_start|>assistant
```

The model sees `<|im_start|>assistant` at the end and knows it needs to generate the reply.

### What is a system prompt?

A system prompt is an instruction given **before** the conversation starts. It sets the model's persona, role, or constraints. In the example above, "You are a pirate" caused the model to respond in pirate speech because the model learned from training that system prompts like that change its behavior.

The system prompt is invisible to the end user in most applications — only the developer sets it. In qwen3.c you set it interactively, which is great for learning.

```{mermaid}
flowchart TD
    SP["system_prompt\ne.g. 'You are a pirate'"]
    UP["user_prompt\ne.g. 'What is the weather?'"]
    SP --> TPL["sprintf into rendered_prompt\nusing ChatML template"]
    UP --> TPL
    TPL --> RP["rendered_prompt:\n<|im_start|>system\nYou are a pirate.<|im_end|>\n<|im_start|>user\nWhat is the weather?<|im_end|>\n<|im_start|>assistant\n"]
    RP --> ENC["encode() → token IDs"]
```

### Suppressing chain-of-thought

Qwen3 has a "thinking" mode where it shows its reasoning in `<think>...</think>` blocks. When thinking is **off** (the default, `-k 0`), the code injects an **empty think block** to skip the reasoning phase:

```c
// run.c line 970
if (!think_on) {
    strcat(rendered_prompt, "<think>\n\n</think>\n");
}
```

This appends an empty reasoning block, so the model skips it and goes straight to the answer. When you run with `-k 1`, this line is skipped and the model produces reasoning first.

### Try it yourself

Run the model and experiment with different system prompts:

```bash
./run Qwen3-0.6B-FP32.gguf
# Try these system prompts:
# "You are a helpful assistant that only answers in bullet points."
# "You are a teacher explaining things to a 10-year-old."
# "Always respond in exactly one sentence."
# (press Enter to skip) — no system prompt
```

Notice how dramatically the behavior changes with the same question.

### Check your understanding

- What problem does a chat template solve?
- What would happen if you did not use the ChatML format?
- Why is the system prompt text included before the user prompt, not after?

---

## Lesson 4: Tokenization — BPE in Action

Now that we know text gets wrapped in a chat template, let us see exactly how that template string becomes integer IDs the model can process.

This is done by the `encode()` function in `run.c` (line 646).

### Step-by-step: how BPE works

**The core idea:** Start with individual characters, then greedily merge the most common adjacent pairs.

During training, Qwen3 was trained with a tokenizer that learned 151,386 merge rules. These rules are stored in `merges.txt`.

Here is a simplified example:

```
Input: "hello"

Step 1: Split into characters (using byte-to-unicode mapping):
["h", "e", "l", "l", "o"]

Step 2: Check all adjacent pairs against merge rules:
- "h" + "e" → merge rank 423 (merge!)
- "he" + "l" → merge rank 891 (merge!)
- "hel" + "l" → merge rank 1203 (merge!)
- "hell" + "o" → merge rank 2891 (merge!)
→ ["hello"]

Step 3: Look up "hello" in vocab → token ID 15339
```

### The encode() function in C

The three phases of `encode()` (lines 646–728):

**Phase 1 — Character split:**

```c
const char *p = rendered_prompt;
while (*p) {
    int match_len = 0;
    // Check if this is a special token like <|im_start|>
    int special_id = match_special_token(p, &match_len);
    if (special_id >= 0) {
        tokens[count++] = strdup(vocab[special_id]);
        p += match_len;
        continue;
    }
    // Otherwise convert raw byte to its unicode representation
    unsigned char b = *p++;
    tokens[count++] = strdup(unicode_bytes[b]);
}
```

**Phase 2 — BPE merge loop:**

```c
bool changed = true;
while (changed) {
    int best_rank = INT_MAX;
    int best_pos = -1;
    // Find the pair with the lowest (highest-priority) merge rank
    for (int i = 0; i < count - 1; i++) {
        int rank = get_merge_rank(tokens[i], tokens[i + 1]);
        if (rank < best_rank) {
            best_rank = rank;
            best_pos = i;
        }
    }
    if (best_pos == -1) break;
    // Merge the best pair in-place
    char *merged = malloc(MAX_TOKEN_LEN * 2);
    snprintf(merged, MAX_TOKEN_LEN*2, "%s%s", tokens[best_pos], tokens[best_pos+1]);
    tokens[best_pos] = merged;
    count--;  // one fewer token after merge
}
```

**Phase 3 — Map strings to IDs:**

```c
for (int i = 0; i < count; i++) {
    for (int j = 0; j < 151936; j++) {
        if (strcmp(tokens[i], vocab[j]) == 0) {
            token_ids[token_id_count++] = j;  // j IS the token ID
            break;
        }
    }
}
```

### Byte-to-Unicode: why it matters

There is a step before BPE: each raw byte is mapped to a Unicode character. This GPT-2 convention ensures every possible byte value (0–255) has a unique printable representation, so the tokenizer can handle any text — including non-English characters and raw bytes.

```c
// run.c line 566
void init_byte_unicode_map() {
    // Printable ASCII (33–126) maps to itself
    // Bytes 0–32 and 127–255 map to codepoints starting at 256
    // This means every byte has a valid token in the vocab
}
```

### The encode call

The whole flow is triggered by a single call (line 975):

```c
encode(tokenizer, rendered_prompt, prompt_tokens, &num_prompt_tokens, multi_turn);
```

After this call, `prompt_tokens[]` contains the integer IDs ready to feed into the transformer.

### Try it yourself (add a debug print)

Open `run.c`, find line 975, and add a temporary debug print right after:

```c
encode(tokenizer, rendered_prompt, prompt_tokens, &num_prompt_tokens, multi_turn);

// ADD THIS:
printf("Token IDs: ");
for (int i = 0; i < num_prompt_tokens; i++) {
    printf("%d ", prompt_tokens[i]);
}
printf("\n");
```

Recompile (`make run`) and run a short prompt. You will see the raw integers the model receives.

### Check your understanding

- What are the three phases of `encode()`?
- Why do special tokens like `<|im_start|>` get handled separately in Phase 1?
- What does a "merge rule" say, and where is it stored?

---

## Lesson 5: The Transformer — How the Model Thinks

The transformer is the core of the LLM. For each token, it computes a probability distribution over all 151,936 possible next tokens.

### The mental model

Think of the transformer as a very deep stack of "refinement stages". A raw token embedding goes in at the bottom (a vector of 1024 numbers). It passes through 28 layers. By the end, the vector has been refined to encode rich contextual meaning — influenced by every other token in the conversation.

```{mermaid}
flowchart TD
    TOK["Token ID\ne.g. 9906 = 'Hello'"] --> EMB["Embedding lookup\nrow 9906 of the table\n→ 1024 floats"]
    EMB --> L1["Layer 1\nAttention + FFN"]
    L1 --> L2["Layer 2\nAttention + FFN"]
    L2 --> DOT["..."]
    DOT --> L28["Layer 28\nAttention + FFN"]
    L28 --> RN["Final RMSNorm"]
    RN --> LG["matmul → logits\n151936 scores"]
```

### The model config for Qwen3 0.6B

```c
// run.c line 18
Config {
    dim        = 1024,  // size of token embedding vector
    hidden_dim = 3072,  // FFN intermediate size
    n_layers   = 28,    // number of transformer blocks
    n_heads    = 16,    // query attention heads
    n_kv_heads = 8,     // key/value heads (GQA)
    vocab_size = 151936,
    seq_len    = 32768, // max context window (tokens)
    head_dim   = 128,   // dimension per attention head
}
```

### Token embedding: the starting point

```c
// run.c line 309
memcpy(s->x, w->token_embedding_table + token * p->dim, p->dim * sizeof(float));
```

Every token has a learned 1024-dimensional vector representation. Token 9906 ("Hello") → copy 1024 floats starting at row 9906 of the embedding table. This vector `s->x` is the starting state that will be refined across 28 layers.

### What happens in each layer

Each of the 28 layers has two sub-modules:

1. **Multi-head attention** — lets each token "look at" other tokens and incorporate their information
2. **Feed-forward network (FFN)** — independently transforms each token's representation

Both sub-modules use **residual connections**: the input is added back to the output. This prevents gradients from vanishing during training and makes the network easier to optimise.

```{mermaid}
flowchart TD
    X["x (1024 floats)"] --> RN1["RMSNorm"]
    RN1 --> ATT["Multi-head Attention\n(see Lesson 6)"]
    ATT --> ADD1["x = x + attention_output\nresidual connection"]
    ADD1 --> RN2["RMSNorm"]
    RN2 --> FFN["Feed-forward Network\nSwiGLU activation"]
    FFN --> ADD2["x = x + ffn_output\nresidual connection"]
    ADD2 --> XOUT["x (updated, 1024 floats)\n→ next layer"]
```

### RMSNorm: stabilising the signal

Before attention and before FFN, the vector is normalised using **RMSNorm** (Root Mean Square Normalisation):

```c
// run.c line 247
void rmsnorm(float* o, float* x, float* weight, int size) {
    float ss = 0.0f;
    for (int j = 0; j < size; j++) ss += x[j] * x[j];
    ss = 1.0f / sqrtf(ss / size + 1e-6f);
    for (int j = 0; j < size; j++) o[j] = weight[j] * (ss * x[j]);
}
```

This keeps the values in a stable range, which makes training and inference numerically stable.

### FFN: the "knowledge store"

The feed-forward network in each layer is where most of the model's "knowledge" is stored. It uses **SwiGLU** activation — a gated variant of the standard MLP:

```c
// run.c line 406
matmul(s->hb,  s->xb, w->w1, dim, hidden_dim); // up-project (1024 → 3072)
matmul(s->hb2, s->xb, w->w3, dim, hidden_dim); // gate

for (int i = 0; i < hidden_dim; i++) {
    float val = s->hb2[i];
    val *= (1.0f / (1.0f + expf(-val))); // silu: x * sigmoid(x)
    val *= s->hb[i];                      // multiply by up-projected
    s->hb2[i] = val;
}

matmul(s->xb, s->hb2, w->w2, hidden_dim, dim); // down-project (3072 → 1024)
```

The gate controls how much of each feature passes through — giving the network a selective memory.

### Final output

After all 28 layers:

```c
// run.c line 429
rmsnorm(s->x, s->x, w->rms_final_weight, p->dim); // final normalise
matmul(s->logits, s->x, w->wcls, p->dim, p->vocab_size); // project to 151936
```

`s->logits` is now a vector of 151,936 floats — one score per possible next token.

### Check your understanding

- What is the shape of the token embedding vector for Qwen3 0.6B?
- What does a residual connection do, and why is it important?
- After all 28 layers, what does the output vector represent?

---

## Lesson 6: Attention — How Tokens Talk to Each Other

Attention is the most important innovation in transformers. It solves a fundamental problem: when predicting the next token, the model needs to look back at relevant parts of the context — not just the token immediately before.

### The intuition

Imagine reading the sentence: "The trophy didn't fit in the suitcase because **it** was too big."

When you read "it", you need to look back and figure out what "it" refers to. Is it the trophy or the suitcase? Humans do this automatically. Attention is the mechanism that lets the model do the same.

Each token produces three vectors:
- **Query (Q):** "What am I looking for?"
- **Key (K):** "What do I contain?"
- **Value (V):** "What information should I pass on?"

Attention scores are computed by matching a token's Q against all other tokens' Ks. Tokens with high Q·K scores are attended to more.

```{mermaid}
flowchart TD
    T1["Token: 'trophy'"] --> K1["Key K1"]
    T2["Token: 'suitcase'"] --> K2["Key K2"]
    T3["Token: 'it'"] --> Q3["Query Q3"]
    Q3 --> S1["Score = Q3·K1\n(how much 'it' attends to 'trophy')"]
    Q3 --> S2["Score = Q3·K2\n(how much 'it' attends to 'suitcase')"]
    S1 --> SF["softmax → attention weights"]
    S2 --> SF
    SF --> OUT["Weighted sum of Values\n= contextualised representation of 'it'"]
```

### Multi-head attention

Instead of one set of Q/K/V, Qwen3 uses **16 attention heads** in parallel. Each head attends to different aspects of the context (syntax, semantics, co-reference, etc.). The results are concatenated and projected back.

### Grouped Query Attention (GQA)

A memory optimisation: Qwen3 has 16 query heads but only **8 key/value heads**. Each KV head is shared by 2 Q heads. This halves the KV cache size.

```c
// run.c line 303
int kv_mul = p->n_heads / p->n_kv_heads;  // = 2, each KV head serves 2 Q heads

// Accessing the right KV head for query head h:
float* k = s->key_cache + loff + t * kv_dim + (h / kv_mul) * p->head_dim;
```

### RoPE: positional encoding

The model needs to know the *position* of each token — is "bank" the 3rd word or the 30th? Qwen3 uses **Rotary Position Encoding (RoPE)**, which encodes position by rotating the Q and K vectors.

```c
// run.c line 340
for (int i = 0; i < p->head_dim/2; i++) {
    float freq = 1.0f / powf(1000000.0f, (float)i / (p->head_dim/2));
    float fcr = cosf(pos * freq);  // position-dependent rotation
    float fci = sinf(pos * freq);
    // Rotate query
    q[i]             = x_q * fcr - y_q * fci;
    q[i + head_dim/2] = x_q * fci + y_q * fcr;
}
```

Two tokens close in position have similar rotations, so their Q·K scores reflect proximity naturally.

### The KV cache

Every time `forward()` is called for a new position `pos`, it computes Q, K, V for that token and **stores K and V in a cache**:

```c
// run.c line 315
s->k = s->key_cache + loff + pos * kv_dim;  // write K at position pos
s->v = s->value_cache + loff + pos * kv_dim; // write V at position pos
```

At the next step, the new token's Q attends to **all stored Ks and Vs** — not just the current one. This means past tokens are never recomputed, making inference much faster.

```{mermaid}
flowchart LR
    subgraph gen1 ["Step 1: token 'The'"]
        K1["K1 stored"]
        V1["V1 stored"]
    end
    subgraph gen2 ["Step 2: token 'cat'"]
        K2["K2 stored"]
        Q2["Q2 attends to K1, K2"]
    end
    subgraph gen3 ["Step 3: token 'sat'"]
        K3["K3 stored"]
        Q3["Q3 attends to K1, K2, K3"]
    end
    K1 --> Q2
    K1 --> Q3
    K2 --> Q3
```

### Attention score computation

```c
// run.c line 370
for (int t = 0; t <= pos; t++) {
    float* k = s->key_cache + loff + t * kv_dim + (h / kv_mul) * p->head_dim;
    float score = 0;
    for (int i = 0; i < p->head_dim; i++)
        score += q[i] * k[i];               // dot product
    att[t] = score / sqrtf(p->head_dim);    // scale by sqrt(d_k)
}
softmax(att, pos + 1);                       // normalise to probabilities
```

The scaling by `sqrt(head_dim)` prevents scores from being too large before softmax, which would make gradients vanish.

### Check your understanding

- What do Q, K, and V represent conceptually?
- Why does Qwen3 use 8 KV heads instead of 16?
- What is the KV cache and why does it make inference faster?
- What would happen without positional encoding (RoPE)?

---

## Lesson 7: Temperature and Sampling — Controlling Creativity

After `forward()` produces 151,936 logit scores, we need to pick the next token. How we do this dramatically affects the output.

### Three strategies

```{mermaid}
flowchart TD
    L["logits[151936]\nraw scores from transformer"] --> T["÷ temperature"]
    T --> SM["softmax → probabilities"]
    SM --> S1{sampling strategy}
    S1 -- "temperature=0\ngreedy" --> ARG["argmax: always pick\nhighest probability"]
    S1 -- "top-p=0.95\ndefault" --> TP["nucleus: keep\nsmallest set ≥ 95% mass"]
    TP --> RND["sample randomly\nfrom filtered set"]
    S1 -- "temperature=1.5\nrandom" --> FULL["sample from\nfull distribution"]
    ARG --> NEXT["next token ID"]
    RND --> NEXT
    FULL --> NEXT
```

### Temperature

Temperature is the single most important knob for controlling model behavior.

```c
// run.c: inside sample()
for (int q = 0; q < n; q++) {
    logits[q] /= temperature;   // divide all scores
}
softmax(logits, n);              // convert to probabilities
```

| Temperature | Effect | Use case |
|------------|--------|----------|
| 0.0 | Always picks the most likely token. Fully deterministic. | Factual answers, code |
| 0.6 (default) | Slight randomness. Coherent and varied. | General chat |
| 1.0 | Raw model probabilities. More creative. | Creative writing |
| 1.5+ | High randomness. Often incoherent. | Extreme creativity / gibberish |

**Intuition:** At temperature 2.0, the distribution becomes nearly flat — all tokens look equally probable. At temperature 0.1, the gap between top and second token is amplified — the model becomes overconfident.

Try this experiment:

```bash
# Run three times with the same prompt, same seed, different temps:
./run Qwen3-0.6B-FP32.gguf -t 0.1 -s 42
Q: Write a one sentence story about a cat.

./run Qwen3-0.6B-FP32.gguf -t 0.6 -s 42
Q: Write a one sentence story about a cat.

./run Qwen3-0.6B-FP32.gguf -t 1.5 -s 42
Q: Write a one sentence story about a cat.
```

The `-s 42` sets the random seed, making results reproducible.

### Top-p (Nucleus) Sampling

Even after temperature, the distribution may have many low-probability tokens. Top-p sampling keeps only the **smallest set of tokens whose cumulative probability meets or exceeds p**.

Default: `top-p = 0.95`. This means: sort tokens by probability descending, keep adding until you have covered 95% of the total probability mass, sample only from those.

```c
// run.c line 820 (sample_topp)
// Sort probabilities descending
// Find cutoff index where cumsum >= topp
// Sample uniformly from the top candidates
```

This prevents the model from ever generating very rare tokens — even at high temperatures.

### argmax (temperature = 0)

When temperature is 0, just pick the highest score:

```c
// run.c line 780
int sample_argmax(float *probabilities, int n) {
    int max_i = 0;
    float max_p = probabilities[0];
    for (int i = 1; i < n; i++) {
        if (probabilities[i] > max_p) { max_i = i; max_p = probabilities[i]; }
    }
    return max_i;
}
```

This is fully deterministic — same input always produces same output.

### Check your understanding

- What does temperature 0 produce, and why is it called "greedy"?
- If the top token has 60% probability at default temperature, what happens to its probability at temperature 2.0?
- Why does top-p sampling help at high temperatures?

---

## Lesson 8: The Chat Loop — Everything Working Together

Now we can trace the complete `chat()` function (lines 927–1037) — where all the pieces come together.

### The loop structure

```{mermaid}
stateDiagram-v2
    [*] --> UserTurn
    UserTurn --> ReadSystemPrompt : first turn only
    ReadSystemPrompt --> ReadUserPrompt
    UserTurn --> ReadUserPrompt : subsequent turns
    ReadUserPrompt --> Exit : blank input
    Exit --> [*]
    ReadUserPrompt --> BuildTemplate : has input
    BuildTemplate --> Encode : ChatML string
    Encode --> Prefill : feed prompt tokens
    Prefill --> Generate : last prompt token done
    Generate --> Decode : new token
    Decode --> EOS : token == 151645
    EOS --> UserTurn : start next turn
    Decode --> Generate : not EOS
```

### Prefill vs. generation

The loop variable `pos` tracks the current position in the sequence. Every step calls `forward(transformer, token, pos)` but the `token` comes from different sources:

```c
// run.c line 988
if (pos < (multi_turn ? tb->size : num_prompt_tokens)) {
    // PREFILL: feed prompt tokens one by one
    token = (multi_turn) ? tb->data[pos] : prompt_tokens[pos];
} else {
    // GENERATION: feed our own last output
    token = next;
}

float* logits = forward(transformer, token, pos);
next = sample(sampler, logits);
pos++;
```

**Prefill:** The model processes the prompt. Fast — no tokens are generated yet. The KV cache fills up with keys and values for each prompt position.

**Generation:** The model outputs the next token, which is fed back as input. One `forward()` call per generated token. Slower.

### Autoregressive generation

This is the key concept: **the model feeds its own output back as its next input**.

```
forward("What") → logits → sample → "is"
forward("is")   → logits → sample → "the"
forward("the")  → logits → sample → "capital"
...
```

Each call attends to all previous positions via the KV cache. The model has full context without recomputing past tokens.

### EOS detection

The model stops when it generates the end-of-sequence token:

```c
// run.c line 1007
if (next == 151645) {  // <|im_end|> token ID
    printf("\n");
    user_turn = 1;     // back to user input
}
```

### Streaming output

Notice the `fflush(stdout)` after each token:

```c
// run.c line 1025
char *decoded = decode_token_id(next);
printf("%s", decoded);
fflush(stdout);  // flush immediately — this is the "streaming" effect
free(decoded);
```

This is why text appears word-by-word in real time rather than all at once.

### Multi-turn and prefix caching

With `-m 1`, all tokens are accumulated in a `TokenBuffer`:

```c
// run.c line 979
append_tokens(tb, prompt_tokens, num_prompt_tokens);
```

On the next turn, the model processes the full history. The Sep-2025 update added **prefix caching** — the TTFT (time to first token) stays consistent regardless of conversation length, because past prefixes are cached.

### Check your understanding

- What is the difference between the prefill and generation phases?
- Why is `fflush(stdout)` necessary for the streaming effect?
- How does the model "remember" earlier parts of the conversation?

---

## Lesson 9: Loading the Model — GGUF and mmap

One piece we have not explained yet: how does the 3 GB file get into memory so fast?

### GGUF file format

GGUF (GGML Unified Format) is the file format used by llama.cpp, qwen3.c, and many local inference tools. It stores:

- **Header:** model config (architecture, dimensions, vocab)
- **Tensor data:** raw weight arrays in the specified precision (FP32 in our case)

### Memory mapping

Rather than reading the file into RAM, qwen3.c uses `mmap` — the OS maps the file directly into the process's virtual address space:

```c
// run.c line 171
*data = mmap(NULL, *file_size, PROT_READ, MAP_PRIVATE, *fd, 0);
void* weights_ptr = ((char*)*data) + 5951648; // skip header
memory_map_weights(weights, config, weights_ptr);
```

The weight pointers (`wq`, `wk`, `wv`, etc.) are just offsets into this memory-mapped region. When the model accesses them, the OS reads the relevant pages from disk on demand. This is why:

- The model "starts" immediately even though the file is 3 GB
- First inference is slower (disk reads); subsequent runs are faster (OS page cache)

```{mermaid}
flowchart LR
    subgraph disk ["Disk (3 GB GGUF)"]
        H["Header\n5.9 MB"]
        W["Weights\n~2.4 GB"]
    end
    subgraph virt ["Virtual Memory (mmap)"]
        PTR["float* pointers\nzero-copy"]
    end
    subgraph code ["C code"]
        WQ["w->wq"]
        WK["w->wk"]
        WV["w->wv"]
    end
    disk -- "mmap()" --> virt
    virt --> code
```

### Config from header.txt

The architecture parameters (dim, n_layers, etc.) are loaded from `header.txt` — a text file extracted from the GGUF by `extract_v_m.py`:

```c
// run.c line 218
if (strcmp(key, "QWEN3_EMBEDDING_LENGTH") == 0)
    t->config.dim = atoi(val);
else if (strcmp(key, "QWEN3_BLOCK_COUNT") == 0)
    t->config.n_layers = atoi(val);
// ...
```

---

## The Complete Picture

Putting every lesson together into one diagram:

```{mermaid}
flowchart TD
    subgraph startup ["Startup — runs once"]
        G["GGUF file (3 GB)"] -- "mmap()" --> TW["TransformerWeights\n(pointers, zero-copy)"]
        HT["header.txt"] --> CFG["Config\ndim=1024, layers=28..."]
        VT["vocab.txt"] --> VOC["vocab[151936]"]
        MT["merges.txt"] --> MRG["merges[151386] BPE rules"]
    end

    subgraph turn ["Each conversation turn"]
        UI["User types message"] --> SYS["+ System Prompt"]
        SYS --> TPL["ChatML Template\n<|im_start|>system...<|im_end|>\n<|im_start|>user...<|im_end|>\n<|im_start|>assistant\n"]
        TPL --> ENC["encode()\nBPE merge → token IDs"]
        ENC --> PF["Prefill phase\nforward() × prompt_length\nfills KV cache"]
        PF --> GEN["Generation phase\nforward() × 1 → sample → next token"]
        GEN --> DEC["decode_token_id()"]
        DEC --> PRINT["printf + fflush\nstream to screen"]
        PRINT --> EOS{next == 151645?}
        EOS -- No --> GEN
        EOS -- Yes --> UI
    end

    startup --> turn
```

---

## Learning Plan for First Break AI — Step 2

This plan maps to the [First Break AI Step 2 roadmap](https://thefirehacker.github.io/firstbreakai/roadmap.html).

### Concept-to-Code Mapping

| Concept | Covered In | Code Location | Exercise |
|---------|-----------|---------------|---------|
| What is an LLM? | Lesson 0 | Conceptual | Read, no code yet |
| Running a model | Lesson 1 | `make run`, `./run` | Run with 3 different system prompts |
| Tokens + vocabulary | Lesson 2 | `run.c` 451–523 | Print token IDs (debug exercise) |
| Chat templates | Lesson 3 | `run.c` 962–972 | Try 5 different system prompts |
| BPE tokenization | Lesson 4 | `run.c` 646–728 | Add print to encode(), trace a word |
| Transformer architecture | Lesson 5 | `run.c` 297–433 | Count layers, read Config struct |
| Attention + KV cache | Lesson 6 | `run.c` 360–395 | Trace attention score computation |
| Temperature + sampling | Lesson 7 | `run.c` 763–902 | Compare `-t 0.1` vs `-t 1.5` |
| Chat loop | Lesson 8 | `run.c` 927–1037 | Add token ID prints |
| GGUF + mmap | Lesson 9 | `run.c` 158–184 | Check `header.txt` values |

---

### Phase 1: Get It Running (Day 1)

**Goal:** Model running, basic intuition about what an LLM is.

**Theory:**
- [ ] Read Lesson 0 (What is an LLM?)
- [ ] Understand: "next token prediction is all it does"
- [ ] Understand: "parameters = numbers learned during training"

**Practice:**
- [ ] Run `./run Qwen3-0.6B-FP32.gguf` — basic chat
- [ ] Run with reasoning: `./run ... -k 1`
- [ ] Run with multi-turn: `./run ... -m 1`
- [ ] Experiment with 5 different system prompts

**Verification:**
- [ ] I can explain what an LLM is to someone with no CS background
- [ ] I ran the model successfully on my Mac
- [ ] I observed the difference a system prompt makes

---

### Phase 2: Tokens and Chat Templates (Day 2)

**Goal:** Understand how text becomes numbers and how the model knows it's a "chat".

**Theory:**
- [ ] Read Lesson 2 (Tokens)
- [ ] Read Lesson 3 (Chat Templates)
- [ ] Understand: what is a token ID?
- [ ] Understand: why does the ChatML format exist?

**Practice:**
- [ ] Open `run.c` and find the ChatML template (line 963)
- [ ] Find the special tokens array (line 476)
- [ ] Add debug print after `encode()` call (line 975) to see token IDs
- [ ] Run with `-k 0` vs `-k 1` and compare the `<think>` injection

**Verification:**
- [ ] I can explain what `<|im_start|>` is and why the model needs it
- [ ] I have seen the raw token IDs printed
- [ ] I can trace the path from "user types text" to "prompt_tokens[] array"

---

### Phase 3: Tokenization Deep Dive (Day 3)

**Goal:** Understand BPE — how the merger rules work.

**Theory:**
- [ ] Read Lesson 4 (BPE encoding)
- [ ] Understand: what is a merge rule?
- [ ] Understand: what is the byte-to-unicode mapping?

**Practice:**
- [ ] Open `merges.txt` and read 10 entries — what do you see?
- [ ] Open `vocab.txt` and look at entries 0–50 and around 9906 — what patterns do you see?
- [ ] Add prints inside the BPE merge loop (Phase 2 of `encode()`) to see tokens merge
- [ ] Try to tokenize "tokenization", "qwen3", and an emoji manually by following the code

**Verification:**
- [ ] I can trace the 3 phases of `encode()` in plain English
- [ ] I understand why special tokens are handled before BPE

---

### Phase 4: Attention and the Transformer (Day 4–5)

**Goal:** Understand how the transformer processes tokens through attention and FFN layers.

**Theory:**
- [ ] Read Lesson 5 (Transformer)
- [ ] Read Lesson 6 (Attention)
- [ ] Understand: Q, K, V and how attention scores are computed
- [ ] Understand: what is the KV cache and why it makes generation faster?

**Practice:**
- [ ] Read the `forward()` function start-to-end (lines 297–433)
- [ ] Find the attention score loop (line 370) and trace it for head 0
- [ ] Find the KV cache write (lines 315–316) and the read (line 371)
- [ ] Calculate: how much RAM does the KV cache use?  
  Formula: `2 × n_layers × seq_len × n_kv_heads × head_dim × 4 bytes`
- [ ] Find the FFN (lines 406–421) and identify which matrix is "up", "gate", and "down"

**Verification:**
- [ ] I can draw the attention mechanism from memory (Q, K, V → scores → softmax → weighted sum)
- [ ] I understand why GQA uses 8 KV heads instead of 16
- [ ] I calculated the KV cache size for a 32K context window

---

### Phase 5: Sampling and the Full Loop (Day 6)

**Goal:** Understand temperature, sampling, and how the chat loop generates responses.

**Theory:**
- [ ] Read Lesson 7 (Temperature + Sampling)
- [ ] Read Lesson 8 (Chat Loop)
- [ ] Understand: what is the difference between prefill and generation?

**Practice:**
- [ ] Temperature experiment: same prompt with `-t 0.1`, `-t 0.6`, `-t 1.5`, `-s 42` (fixed seed)
- [ ] Add prints in the chat loop (line 995) to see `token` and `next` values
- [ ] Run with `-r 1` flag to see tokens/second
- [ ] Try very long multi-turn conversation: does TTFT stay stable? (use `-f 1` flag)

**Verification:**
- [ ] I can explain why temperature 0 is called "greedy"
- [ ] I can trace the full loop from encode() → forward() → sample() → decode() → print
- [ ] I understand why multi-turn is slower than single-turn (without prefix caching)

---

### Exercises for Fast Learners

**Exercise A: Token Counter**

Write a C function that prints how many tokens a given string contains, before the model even starts generating. Call `encode()` and print `num_prompt_tokens`.

**Exercise B: Entropy Measurement**

After `softmax` in the sampler, compute the entropy of the distribution:  
`H = -sum(p * log(p))`.  
High entropy = uncertain model. Low entropy = confident model. Print this with each generated token.

**Exercise C: KV Cache Visualisation**

After each `forward()` call in the generation phase, print the magnitude of the key vector for layer 0:  
`sqrt(sum(k[i]^2 for i in range(head_dim)))`.  
Watch how it changes token by token.

**Exercise D: Greedy vs. Sampling**

Implement a flag `-g 1` for greedy decoding. Compare outputs with the default top-p sampler. Which is "better"?

**Exercise E: Manual Tokenization**

Take a sentence, open `merges.txt` and `vocab.txt`, and manually trace through the BPE algorithm on paper. Does your result match what `encode()` produces?

---

## What's Next — First Break AI Step 3

You have completed Step 2. Here is a map to Step 3:

| What you learned here | What Step 3 adds |
|----------------------|-----------------|
| Running one model (qwen3.c) | Running many models via inference servers (vLLM, llama.cpp server) |
| FP32 (full precision) | Quantization: GGUF Q4, GPTQ, AWQ — smaller, faster |
| Single request, synchronous | Batching: many requests in parallel |
| Chat via stdin | Serving via API (OpenAI-compatible endpoints) |
| Single token at a time | Continuous batching, throughput vs. latency tradeoffs |

The [Step 3 roadmap](https://thefirehacker.github.io/firstbreakai/roadmap.html) covers all of this. Everything you understood here — tokens, attention, KV cache, temperature — is the foundation for understanding why those systems are designed the way they are.

---

## Progress Tracker

Copy and paste this into your own notes. Check off items as you complete them.

```
## My First Break AI Step 2 Progress

### Phase 1: Get It Running
- [ ] Model running on my Mac
- [ ] Tried 5 different system prompts
- [ ] Ran with reasoning mode (-k 1)
- [ ] Ran multi-turn mode (-m 1)

### Phase 2: Tokens and Chat Templates
- [ ] Found ChatML template in run.c
- [ ] Added debug print for token IDs
- [ ] Observed <think> injection for -k 0

### Phase 3: Tokenization Deep Dive
- [ ] Read merges.txt — understood merge rules
- [ ] Added prints inside BPE merge loop
- [ ] Traced encode() for a short string

### Phase 4: Attention and Transformer
- [ ] Read forward() function start to end
- [ ] Found Q, K, V projections in code
- [ ] Calculated KV cache RAM usage

### Phase 5: Sampling and Full Loop
- [ ] Temperature experiment (0.1 vs 1.5)
- [ ] Added token ID prints in chat loop
- [ ] Measured tokens/second with -r 1 flag

### Fast Learner Exercises
- [ ] Exercise A: Token Counter
- [ ] Exercise B: Entropy Measurement
- [ ] Exercise C: KV Cache Visualisation
- [ ] Exercise D: Greedy vs Sampling
- [ ] Exercise E: Manual Tokenization

### Step 2 Complete?
- [ ] I can explain: what is a token?
- [ ] I can explain: what is a chat template?
- [ ] I can explain: how does attention work?
- [ ] I can explain: what does temperature do?
- [ ] I can explain: what is the KV cache?
- [ ] I ran the model and experimented with it
- [ ] Ready for Step 3: Inference deep dive
```

---

## Summary

In one post, starting from zero:

| Concept | What it is | Where in code |
|---------|-----------|---------------|
| Token | Chunk of text, represented as an integer | `vocab.txt`, `run.c:491` |
| Vocabulary | 151,936 known tokens | `run.c:459` |
| BPE | Algorithm to split text into tokens | `run.c:646` |
| Chat template | ChatML format marking speaker turns | `run.c:963` |
| System prompt | Pre-conversation instruction to model | `run.c:951` |
| Token embedding | Vector representation of a token | `run.c:309` |
| Attention | Mechanism for tokens to exchange info | `run.c:360` |
| KV cache | Memory of past keys/values | `run.c:315` |
| RoPE | Encodes token position via rotation | `run.c:340` |
| Temperature | Controls randomness of output | `run.c:sample()` |
| Top-p | Nucleus sampling — filters long tail | `run.c:820` |
| Prefill | Processing the prompt (fast) | `run.c:988` |
| Generation | Autoregressive next-token loop (slow) | `run.c:991` |
| EOS | End-of-sequence token (151645) | `run.c:1007` |
| GGUF + mmap | Zero-copy model loading | `run.c:171` |

All of this in a single C file, 1130 lines, no external dependencies. The best way to learn how inference works is to read code like this — and now you have.

Join the [First Break AI Discord](https://thefirehacker.github.io/firstbreakai/) to share what you built and get help on the exercises.
