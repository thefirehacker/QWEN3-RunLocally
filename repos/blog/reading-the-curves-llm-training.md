---
title: "Reading the Curves: How Real LLMs Learn, Spike, Recover, and Stabilize"
description: "A practical guide to LLM training graphs using NanoGPT, OLMo, and Marin—Weights & Biases panels, loss shapes, data mix, good vs bad spikes, and a deep-dive case study on the Marin 32B QK-Norm warm-start."
date: 2026-04-25
categories: [llm, training, weights-biases, olmo, nanogpt, marin, first-break-ai]
---

> **First Break AI — Step 4: Training fundamentals**
>
> This post is part of the [First Break AI](https://cohort.bubblnet.com/) cohort roadmap.
> Step 4 is where you move from *running* models to *training* and *reading* real runs: data pipelines, distributed training, and experiment tracking.
> If you are coming from a modded nanoGPT or Keller Jordan speedrun context, you already know what a **val loss** line *feels* like.
> This post is about reading **industrial** runs the way teams do: multiple panels, multiple clocks, and primary sources.

---

# Reading the Curves: How Real LLMs Learn, Spike, Recover, and Stabilize

By the end of this post you should be able to:

- Pick **x-axis units** (steps vs tokens vs time) and know when each misleads
- Read a **data mix table** and understand why mixture composition is the first decision in pretraining
- Open a W&B run and read **loss next to** LR, grad/update norms, and throughput — metric by metric
- Distinguish a **spiky but recoverable** run from a **worse-new-plateau** failure mode
- Load **two OLMo checkpoints** in Python, compare weights and inference outputs, and understand what **2T tokens** of optimization changes
- Walk through the **Marin 32B** story with real numbers: architecture, data mix, mitigations, failed recovery, QK-Norm, warm-start, and benchmark results
- Explain the **"boring is beautiful"** principle: why smooth, featureless loss curves signal healthy training

> **If you only read one part:** skim the [Data pipeline](#lesson-3-data-pipeline-and-data-mix), the [W&B checklists](#lesson-5-weights--biases-is-this-run-healthy), and the [Marin case study](#lesson-9-case-study-marin-32b-the-full-timeline). That trio is the closest thing to a public "loss spike debugging course."

---

## Navigate by roadmap

| Step | Topic | This blog |
|------|-------|-----------|
| Step 1 | First use of AI for coding | — |
| [Step 2](qwen3.c.md) | Run a model locally | [Qwen3 in C](qwen3.c.md) |
| Step 3 | Inference deep dive | — |
| **Step 4** | **Training fundamentals** | **You are here** |
| Step 5+ | Product / capstone | — |

---

## Table of contents

1. [Lesson 1: Loss is a fingerprint](#lesson-1-loss-is-a-fingerprint-not-a-trophy)
2. [Lesson 2: Axes and units](#lesson-2-axes-and-units-steps-tokens-wall-clock)
3. [Lesson 2b: Loss shapes](#lesson-2b-loss-shapes--power-laws-tails-staircases-lr-crossover)
4. [Lesson 3: Data pipeline and data mix](#lesson-3-data-pipeline-and-data-mix)
5. [Lesson 4: NanoGPT speedrun vs production](#lesson-4-after-a-nanogpt-speedrun-small-lab-vs-production)
6. [Lesson 5: Weights & Biases](#lesson-5-weights--biases-is-this-run-healthy)
7. [Lesson 6: Good spike vs bad spike](#lesson-6-good-spike-vs-bad-spike)
8. [Lesson 7: Perfetto and systems traces](#lesson-7-perfetto-systems-level-exercise)
9. [Lesson 7b: Resource map](#lesson-7b-resource-map--where-to-learn-next)
10. [Lesson 8: OLMo, Dolma, checkpoints as time](#lesson-8-olmo-dolma-and-checkpoints-as-time)
11. [Lesson 9: Marin 32B case study](#lesson-9-case-study-marin-32b-the-full-timeline) ← **expanded**
12. [Lesson 10: Qwen3 pretraining stages](#lesson-10-qwen3-pretraining-stages)
13. [Lesson 11: Boring is beautiful](#lesson-11-boring-is-beautiful)
14. [Takeaways and exercises](#takeaways-and-exercises)

---

## Lesson 1: Loss is a fingerprint, not a trophy

Training loss (usually cross-entropy on next-token prediction) is *not* a game score. It is a **fingerprint** of the whole stack working together:

| Layer | What it affects on the curve |
|-------|------------------------------|
| **Data** | Mixture, ordering, quality filters, dataloader bugs |
| **Optimizer** | AdamW moments, clipping, outlier batches |
| **Schedule** | Warmup, hold, decay, mid-run surgery |
| **Architecture** | Depth, GQA, normalization — including **QK-Norm** |
| **Systems** | Compile, comms, checkpoint pauses, grad accumulation |

If you only log a single scalar `loss` every step, you are flying with one instrument. Real teams also log **gradient norm**, **update norm**, and often **per-layer** or **eval** lines — things that *predict* whether loss is about to spike.

### A concrete misread

Imagine two runs on W&B:

- **Run A:** train loss drops smoothly from 3.2 → 2.9 over 50k steps
- **Run B:** train loss is noisier, currently at 2.95

Run A looks better — until you open the **throughput** panel and see Run A's tokens/s dropped to zero for the last 10k steps because the dataloader stalled. Run B is actually learning; Run A is plotting stale or cached numbers.

**Rule:** never crown a run from loss alone. Always read loss as one panel in a **set**.

### Check your understanding

- Name three non-data reasons train loss can look "good" while the run is broken.
- Why is val loss often more trustworthy than train loss for "is this model getting better?"

---

## Lesson 2: Axes and units — steps, tokens, wall clock

Three clocks measure different things. Mixing them up is the most common mistake when comparing runs on Discord or in a paper figure.

| Axis | Definition | Fails when |
|------|------------|------------|
| **Optimizer step** | One `optimizer.step()` after forward/backward | You change **global batch** or **grad accumulation** between runs |
| **Tokens trained** | Cumulative tokens seen in the current phase | Batch schedule changes tokens/step mid-run |
| **Wall clock** | Real time including I/O, eval, checkpoint | Tells you **cost**, not model quality |

### Mental model

**Steps** = your optimizer's heartbeat.
**Tokens** = how much the model has read.
**Wall clock** = how much you paid.

Never compare runs on steps alone unless batch size is identical.

### `train_time_ms` vs `step_avg_ms`

These show up in speedrun and research dashboards:

- **`train_time_ms` (cumulative)** — should grow roughly linearly with step count. A bend or flat region can mean: eval windows, checkpointing, dataloader stall, or a stuck DDP rank.
- **`step_avg_ms` (per step)** — kernels + Python + I/O. A tall spike at the start is often **compile** / **torch.compile** warmup. A slow rise over hours can be memory pressure or contention.

### Which axis when?

| Your question | Use this x-axis | Why |
|---------------|-----------------|-----|
| "Is this run learning at all?" | **Steps** | Fastest feedback — one point per update |
| "How data-efficient vs another model?" | **Tokens** | Normalizes batch size and accumulation |
| "How much did this cost?" | **Wall clock** | GPU-hours = dollars |
| "Did hardware change mid-run?" | **Tokens AND wall clock** | Marin switched TPU v5p-512 → v4-2048 mid-training |

### When the loss line lies

Before you diagnose "the model can't learn," check:

1. **Logging frequency** — raw step loss vs EMA/smoothed
2. **Eval metric** — not accidentally on train shards
3. **Token counter** — in sync with the actual dataloader
4. **DDP** — not plotting rank 0's loss while another rank is stuck

### Check your understanding

- You compare two runs at step 100k. Run A uses batch 512, Run B uses batch 2048. Can you compare loss at step 100k directly?
- Marin switched hardware mid-run. Which two panels do you open first?

---

## Lesson 2b: Loss shapes — power laws, tails, staircases, LR crossover

Smooth decreasing loss is the idealized picture. Real curves have **structure**:

| Shape | What it looks like | What it usually means |
|-------|-------------------|----------------------|
| **Steep early drop** | Fast initial descent | Unigrams, local syntax, easy statistics |
| **Long flat tail** | Last 0.1 nats takes huge token budget | Data quality dominates late pretrain |
| **Staircase** | Step-wise drops | Stage boundary, batch change, or eval overlay artifact |
| **LR crossover** | Two curves cross mid-run | Higher LR wins early, lower LR wins late |

### Three phases (universal shape)

```
loss
  |
10| \
  |  \            Phase 1: steep drop
 5|   \____
  |        \___      Phase 2: slow decline
 3|            \__
  |               \_    Phase 3: flattening
  |____________________ tokens
```

| Phase | What the model learns | Duration (% tokens) |
|-------|----------------------|---------------------|
| Steep drop | Token frequencies, syntax, local patterns | ~5–10% |
| Slow decline | Long-range deps, semantics, early reasoning | ~40–60% |
| Flattening | Rare patterns, refinement | ~30–50% |

*A 124M NanoGPT and a 32B Marin live in different absolute loss ranges — the **shape** transfers.*

### LR crossover trap

```
High_LR:  \___
Low_LR:     \_______   ← can win late

Loss
 | \___________
 +---------------- tokens
     long tail
```

Two runs with different peak learning rates can **cross**: higher LR looks better at 5% of tokens and **loses** at 50%. Documented in OLMo and Qwen3 reports. **Never crown a run from early loss** unless your only goal is early convergence.

### Staircase loss and gradient accumulation

With **gradient accumulation**, one optimizer step spans multiple microbatches. **Per-microbatch** loss looks choppy; **per-step** averages look smoother. When comparing forks, be explicit: which loss is on the plot?

### Check your understanding

- Why does the last 0.1 nats of improvement often cost more tokens than the first 1.0 nats?
- You see a staircase at exactly the S1/S2 boundary in a Qwen3 figure. Bug or feature?

---

## Lesson 3: Data pipeline and data mix

Before you read any curve, you need to know **what the model ate**. Data mix is the **first decision** in pretraining — it determines what the loss curve even *means*.

### The pipeline (each step can break)

```{mermaid}
flowchart TD
    A[Raw web crawl] --> B[Quality filtering]
    B --> C[Deduplication]
    C --> D[Tokenization]
    D --> E[Mixing / weighting by source]
    E --> F[Batching + shuffling]
    F --> G[Dataloader]
    G --> H[Model forward pass]
```

A "learning" curve with a broken dataloader is not learning — it is fitting noise or seeing empty batches.

### Reading a real mix table — Marin Phase 1

From the [Marin 32B retrospective](https://marin-community.github.io/marin/retrospective/32b/), Phase 1–3 pretrain mix:

| Source | Weight (%) | Role |
|--------|------------|------|
| Nemotron-CC (medium quality) | ~30.69 | Broad web coverage |
| Nemotron-CC (HQ synthetic) | ~24.70 | Cleaned, synthetic-augmented web |
| Nemotron-CC (medium-low) | ~13.98 | Lower-tier web |
| Nemotron-CC (HQ actual) | ~8.30 | High-quality non-synthetic |
| Nemotron-CC (other buckets) | ~19.56 combined | Various quality tiers |
| StarCoder | ~2.27 | Code |
| Proofpile 2 | ~0.50 | Math / formal reasoning |

**Total: Nemotron-CC dominates at ~91%.** This is *not* OLMo's Dolma. When you read *any* loss curve, your first question: **what data produced this?**

### Why mix changes between stages

Pretraining is not one monolithic phase:

| Stage | Typical goal | Loss shape |
|-------|--------------|------------|
| **Stage 1 — broad coverage** | Language fundamentals | Steep drop, long slow decline |
| **Stage 2 — reasoning-heavy** | STEM, code, math, synthetic | Slope change at boundary (intentional) |
| **Cooldown / midtraining** | Curated high-quality sources | Small improvements, high data cost |

In Marin's Phase 4 Mantis cooldown (~1.074T tokens), Nemotron-CC dropped from ~91% to ~68%. **MegaMath**, arXiv, finemath, StackExchange, and Wikipedia were added. Standard practice: late training uses targeted data.

### Real pipeline bugs (from Marin and learner forks)

| Bug | Symptom | Fix |
|-----|---------|-----|
| **GSM8k cache contamination** | Math eval inflated | Replace contaminated source with clean MegaMath |
| **LCG shuffle** | Batch-level distribution skew | Feistel shuffle with better mixing |
| **Broken dataloader path** | Loss looks "normal" but eval fails | Decode a batch before first run |

### Sanity check (do before every first run)

```python
batch = next(iter(train_loader))
print(f"Batch shape: {batch.shape}")
print(f"Token range: {batch.min()} to {batch.max()}")
print(f"Sample decode: {tokenizer.decode(batch[0][:50].tolist())}")
# Garbage, all zeros, or empty strings → fix pipeline before reading loss
```

### Check your understanding

- Why does a stage boundary sometimes look like a "spike" on a combined loss plot?
- What three checks prove your dataloader is real before step 1?

---

## Lesson 4: After a NanoGPT speedrun — small lab vs production

A **modded NanoGPT** or Keller-style speedrun is a perfect **microscope**: tight code path, a target **validation loss**, community recipes. The jump to industrial pretrain adds:

- **Data mixture** and **mid-course** schedule changes — documented in reports, not a 200-line `train.py`
- **Stability telemetry** beyond loss: grad norm, update norm, z-loss, MoE router stats
- **Checkpoints as time travel** (OLMo on Hugging Face) — sample the trajectory, not just the endpoint

### Honest failure modes from learners

1. **Fork vendors training but not data construction** — dataloader path points at nothing
2. **DDP traces missing** — cannot see all-reduce bubbles without profiling
3. **Subtle fork bugs** — wrong accumulation count, eval on train split

> **Rule:** if the chart moved but the data pipeline could not have produced that batch, you have a **logging or shard issue**, not a weird model.

### Check your understanding

- What is the one sanity check that catches "silent dataloader" bugs before loss misleads you?

---

## Lesson 5: Weights & Biases — is this run healthy?

Open a new run. Before zooming in on loss, set up a **default panel group**:

| Order | Panel | What you learn |
|-------|-------|----------------|
| 1 | **Held-out loss / perplexity** | Generalization; divergence from train → check leakage |
| 2 | **Train loss** | Optimization fit; can be *too* good vs val |
| 3 | **Gradient norm (pre-clip)** | Leading indicator — spikes *before* loss |
| 4 | **Update norm (post-Adam)** | Actual step size — "was this a wild step?" |
| 5 | **Learning rate + schedule phase** | Loss is not interpretable without schedule context |
| 6 | **Throughput (tokens/s or step_avg_ms)** | Beautiful loss + zero throughput = burning money |
| 7 | **Max grad / clip settings** | When the team turned a knob mid-run |

### Three phases of val loss (indicative)

| Phase | val_loss range | What the model learns | Duration |
|-------|----------------|----------------------|----------|
| Steep drop | ~10→~5 (small) / ~2.7→~2.5 (large) | Token frequencies, syntax | ~5–10% tokens |
| Slow decline | ~5→~3 / ~2.5→~2.35 | Long-range deps, semantics | ~40–60% |
| Flattening | ~3→~2.8 / ~2.35→~2.30 | Rare patterns, refinement | ~30–50% |

*Numbers are illustrative — a 124M NanoGPT and a 32B Marin live in different ranges. The **shape** transfers.*

### The norm pipeline (leading indicators)

```{mermaid}
flowchart LR
    B[Bad batch or instability] --> G[grad_norm spikes]
    G --> A[Adam scaling]
    A --> U[update_norm spikes]
    U --> L[Loss spike next step]
```

By the time loss spikes, damage is often already applied. Norms give you 1–2 steps of warning.

### Red and green flags

| Pattern | Worry about |
|---------|-------------|
| Isolated/repeating **upward** spikes in train loss | Outliers, LR, bad batch, attention numeric issues |
| **Train down, val up** late | Overfit, wrong eval, contamination |
| **Flat** loss early (after warmup) | LR too small, empty data, frozen layers |
| **Norms precede loss** in spikes | Clipping limits damage; may not fix structural issues |

### Train vs val: three useful stories

- **Both down** — happy default for pretrain for a long time (modulo eval quality)
- **Train down, val up** — classic overfit *or* train/eval distribution mismatch *or* eval bug. Check eval construction before calling it overfit
- **Staircase val** — often less frequent eval or EMA artifact; read trend over multiple evals

### Monday checklist (new run)

1. Eval frequency and exact eval split
2. Tokens/step and global batch in config
3. One throughput line
4. Grad or update norm (both if possible)
5. Git SHA and data snapshot id

If five is too many: **(1) eval**, **(2) batch/tokens**, **(3) throughput** on day one; add norms when something looks spiky.

### Check your understanding

- Why is update norm often more informative than grad norm after Adam?
- Train loss down, val loss up — list two bugs *other than* overfit that cause this.

---

## Lesson 6: Good spike vs bad spike

A **spike** is a short **increase** in training loss. Not every spike cancels a run.

**Recoverable (often acceptable):**

```
loss
 |   /\
 |__/  \_____  same band as before
```

**Bad: new, worse plateau:**

```
loss
 |     /\
 |____/  \________  settles higher; trajectory broke
```

### Four questions (every time)

1. On smoothed and unsmoothed train loss, does the run **rejoin** the old trend band?
2. What did **eval** do after the window — *same eval harness*?
3. Do **update norms** and **grad norms** return to a typical band?
4. What **code / data / schedule** event happened at the same step?

### Spike debugging flowchart

```{mermaid}
flowchart TD
    S[Loss spike observed] --> Q1{Did loss return to<br>pre-spike band?}
    Q1 -->|Yes| R1[Recoverable — log and monitor]
    Q1 -->|No| Q2{Did eval loss shift up?}
    Q2 -->|No| R2[Possible logging artifact]
    Q2 -->|Yes| Q3{Did norms return to normal?}
    Q3 -->|Yes| R3[Data event — check shard at that step]
    Q3 -->|No| R4[Structural instability — architecture or LR]
```

### Marin spike timeline (preview — full story in Lesson 9)

| Step | Event | Spikes softened? | Loss recovered? |
|------|-------|------------------|-----------------|
| 0–56k | Periodic spikes, all recovered | — | Yes |
| ~56,400 | max_grad_norm 1.0 → 0.2 | Amplitude down | Partially |
| ~72,233 | Update-norm clipping ON | Further softened | Temporarily |
| ~74k–80k | Update clipping **accidentally off** | Severe spikes return | **No — worse plateau** |
| 80,000 | Optimizer fixes insufficient | — | Architecture change needed |

**Lesson:** three optimizer mitigations *softened* spikes but none *removed* them. Root cause was **attention**, not Adam.

---

## Lesson 7: Perfetto — systems-level exercise

Perfetto shows **time** — GPU kernels, CPU gaps, comms. It does **not** show model quality. It is the plumbing complement to W&B's learning view.

### Capture a trace

```python
from torch.profiler import profile, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    with_stack=True,
    record_shapes=True,
) as prof:
    for step in range(5):
        train_step()

prof.export_chrome_trace("trace.json")
```

Open in [Perfetto UI](https://ui.perfetto.dev/). Find:

1. **Longest GPU kernel** — matmul? all-reduce?
2. **Longest CPU gap** between kernel launches — Python/dataloader overhead
3. **GPU idle > 1ms** — bubble time (sync, launch latency, data starvation)

Relate to W&B: if `step_avg_ms` bumps at step *t*, capture a trace at *t*. Name the stall in one sentence.

> **When traces are missing:** many NanoGPT forks lack profiling. You still have W&B step times, `nvidia-smi`, and per-rank logs — but you cannot see the all-reduce bubble without a capture. Adding profiling is ~10 lines.

### Check your understanding

- Perfetto shows a 3ms GPU idle gap every step. Name two possible causes.
- W&B shows flat loss but rising `step_avg_ms`. Is the model learning?

---

## Lesson 7b: Resource map — where to learn next

| Resource | What it teaches | When to use |
|----------|-----------------|-------------|
| **Smol Training Playbook** | Failure modes, "what we tried" | Intuition for team decisions |
| **UltraScale Playbook** | Distributed systems, throughput | Infrastructure, not model learning |
| **OLMo checkpoints + W&B** | Real loss dynamics, checkpoint evolution | Primary hands-on for this post |
| **Marin 32B retrospective** | Instability postmortem at 32B | Best public debugging story |
| **Qwen3 technical report** | Multi-stage pretrain, QK-Norm | Why modern architectures look this way |
| **Your own NanoGPT run** | End-to-end control | Nothing replaces *your* W&B on *your* data |

You learn most by: **(a)** a tiny run you control end-to-end, and **(b)** one public megaproject where you verify claims against paper + hub.

---

## Lesson 8: OLMo, Dolma, and checkpoints as time

OLMo is **open science**: weights, training code, intermediate checkpoints, W&B groups. **Dolma** is the pretraining corpus — always read the **model card** for the exact build you use.

### Pretraining stages (loss shape changes per stage)

```
[ Stage 1 ] General pretraining on Dolma
   ↓         Loss: steep drop then slow decline
[ Stage 2 ] Midtraining (better quality mix)
   ↓         Loss: staircase at transition (intentional)
[ Stage 3 ] Annealing (curated curriculum)
   ↓         Loss: final refinement
[ Stage 4 ] SFT / RL — out of scope for this post
```

### Hands-on: enumerate checkpoints

```python
from huggingface_hub import list_repo_refs

refs = list_repo_refs("allenai/OLMo-7B-0424-hf")
branches = [b.name for b in refs.branches if b.name.startswith("step")]
print(f"Found {len(branches)} checkpoint branches")
for tag in sorted(branches)[:10]:
    print(f"  {tag}")
```

### Hands-on: compare two checkpoints

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "allenai/OLMo-7B-0424-hf"
tokenizer = AutoTokenizer.from_pretrained(model_id)

early = AutoModelForCausalLM.from_pretrained(
    model_id, revision="step1000-tokens4B", trust_remote_code=True
)
late = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)

prompt = "The capital of France is"
inputs = tokenizer(prompt, return_tensors="pt")

print("Early (4B tokens):")
print(tokenizer.decode(early.generate(**inputs, max_new_tokens=30)[0]))
print("\nLate (2T tokens):")
print(tokenizer.decode(late.generate(**inputs, max_new_tokens=30)[0]))
```

### Hands-on: how much did embeddings change?

```python
import torch

e_early = early.model.embed_tokens.weight.data
e_late  = late.model.embed_tokens.weight.data
diff = (e_late - e_early).norm() / e_early.norm()
print(f"Relative embedding change: {diff:.4f}")
```

You are *feeling* how **2T tokens** changes a fixed probe — not proving benchmark quality.

### Where to read the training story

- **Paper:** [OLMo: Accelerating the Science of Language Models](https://arxiv.org/abs/2402.00838)
- **W&B:** ai2-llm/OLMo-7B pretraining groups — open loss and eval over time
- **Code:** `scripts/train.py` + YAML configs in the OLMo repository

---

## Lesson 9: Case study — Marin 32B (the full timeline)

Everything here is anchored to the official **[Marin 32B Retrospective](https://marin-community.github.io/marin/retrospective/32b/)**. If a forum post disagrees, trust the **report + code + data browser** first.

**Why this story matters:** it is a public, evidence-driven walkthrough of:

```
instability → optimizer mitigations → failed non-architectural recovery → QK-Norm → stabilized long run
```

It is the closest public **postmortem** for a 32B-scale open recipe without joining the lab. Read it like a detective story — each W&B panel is a clue.

**HF base:** [marin-community/marin-32b-base](https://huggingface.co/marin-community/marin-32b-base)

---

### Lesson 9.1 — The setup: what they were trying to build

Marin 32B was a scale-up from a stable **8B "Tootsie"** recipe. The bet: same data mix philosophy, same AdamW schedule, bigger model. **~2.679T tokens** in Phase 1 before the architecture pivot.

#### Architecture (Phase 1 — Llama-style, no QK-Norm)

| Parameter | Value | Notes |
|-----------|-------|-------|
| Hidden size | 5,120 | Main residual stream |
| Intermediate size | 27,648 | FFN expansion (~5.4× hidden) |
| Layers | 64 | Depth |
| Attention heads | 40 | Query heads |
| KV heads | 8 | GQA — 5 queries share each KV group |
| Sequence length | 4,096 | Pretrain context |
| Activation | SiLU | SwiGLU-style FFN |
| Optimizer | AdamW | peak LR **7e-4** |
| Warmup | 1% of steps | |
| Decay window | 40% of steps | |
| Weight decay | 0.05 | |
| EMA beta | 0.995 | |
| Hardware (Phase 1) | TPU v5p-512 | |
| Hardware (Phase 3+) | TPU v4-2048 | Batch adjustments required |

**Scale context:** 64 layers of GQA attention at width 5,120 is where numeric instability showed up — not in the 8B runs that shared the same recipe.

#### Phase overview (from retrospective)

| Phase | Steps (approx.) | Tokens (T) | What changed |
|-------|-----------------|------------|--------------|
| **1** | 0 → 80,000 | 2.679 | Llama-style 32B, no QK-Norm. Spiky loss. |
| **2** | ~80k → ~82k | ≈0.02 (diagnostics) | Necromancy, Muon recovery attempts |
| **3** | 80,000 → 160,000 | 2.684 | Qwen3-style 32B + **QK-Norm**, warm-start from 80k |
| **4+** | 160k → 192k+ | e.g. 1.074 (Mantis cooldown) | Data mix shift, shuffle fixes |

Total in their accounting: on the order of **~6.4T tokens** — see the retrospective for the full breakdown.

---

### Lesson 9.2 — What is QK-Norm and why Marin needed it

If you read [Qwen3 inference Lesson 6](qwen3.c.md#lesson-6-attention--how-tokens-talk-to-each-other), you know attention computes **Q·K** dot products. At large width and depth, those dot products can grow large before softmax — **grad norms spike**, then **loss spikes**.

**QK-Norm** applies RMSNorm to the **Q** and **K** vectors *per head* before the dot product (Qwen3 does this from step 0). It keeps Q and K in a stable numeric range — **stability headroom** at scale.

Marin Phase 1 used a **Llama-style stack without QK-Norm**. The retrospective ties repeated spikes to **attention numeric stress**, not bad data or a broken Adam implementation alone.

| Stack | QK-Norm | Marin phase |
|-------|---------|-------------|
| Llama-style 32B | No | Phase 1 — spiky |
| Qwen3-style 32B | Yes | Phase 3 — smooth |

OLMo 2 and other large-scale reports cite similar motivation — see Marin retrospective references and [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388) Table 1.

---

### Lesson 9.3 — Phase 1: the spiky era (steps 0 → ~80k)

#### What "healthy enough" looked like early (~0 → ~56k)

For roughly the first **56k steps**, training was "as expected" compared to 8B — **except** more frequent **loss spikes** than the smaller model. The team's first diagnostic question was not "why spikes?" but **"do spikes recover without bending the long trajectory?"**

```{mermaid}
flowchart TD
    subgraph wb1 [W&B panels to watch — Phase 1 early]
        L1[train_loss — spiky but recovering]
        G1[grad_norm — spikes BEFORE loss]
        U1[update_norm — follows grad]
        V1[val_loss — still trending down]
    end
    G1 --> L1
    U1 --> L1
    V1 --> OK[Run is stressful but OK]
    L1 --> OK
```

**ASCII — spiky but recoverable:**

```
loss  |  \       /\        /\       /\
      |   \_____/  \______/  \_____/
      +-------------------------------- steps (0 → ~56k)
```

**Teaching point:** a spike is not automatically fatal. Ask: does loss return to the previous band? Does val keep improving?

#### Mitigation 1 — tighten grad clip (~step 56,400)

**Observation:** most grad norms cluster around ~0.2; large norms **precede** spikes.

**Action:** `max_grad_norm` **1.0 → 0.2**

**Effect:** spike **amplitude** softened. Spikes **still appeared**.

| Knob | Controls | Analogy |
|------|----------|---------|
| `max_grad_norm` | Gradient **before** Adam | Limiter on the *input* to the optimizer |

#### Mitigation 2 — update-norm clipping (~step 72,233)

**Action:** clip update norm using rolling mean + 2σ (window 128) — targets **post-Adam** step size.

**Effect:** further softened spikes. Still not sufficient.

| Knob | Controls | Analogy |
|------|----------|---------|
| Update-norm clip | Actual **weight delta** per step | Limiter on the *output* of the optimizer |

#### Mitigation 3 — skip bad steps

Skip updates when update norm is an outlier (OLMo-core / Levanter-style ideas).

**Effect:** softens symptoms; does not fix **structural** attention instability.

#### The bad window (~74k → ~80k)

Update-norm clipping was **accidentally disabled** for a few thousand steps. Severe spikes returned. Loss settled on a **new, worse plateau** — not the old band.

**ASCII — bad plateau (diagnosis changes):**

```
loss  |  \______
      |         \   (old band)
      |          /\
      |         /  \______  ← new WORSE plateau (~74k–80k)
      +-------------------------------- steps
```

```{mermaid}
flowchart TD
    subgraph wb2 [W&B — bad plateau signals]
        L2[train_loss — does NOT rejoin old band]
        V2[val_loss — shifts up]
        N2[grad_norm + update_norm — stay elevated]
    end
    L2 --> BAD[Structural problem — not noise]
    V2 --> BAD
    N2 --> BAD
```

**Teaching point:** recovered spike = warning. **New worse plateau** = diagnosis. Optimizer tricks exhausted → look at **architecture**.

#### Phase 1 spike chronology (single table)

| Step | Event | grad/update norms | Loss trajectory |
|------|-------|-------------------|-----------------|
| 0–56k | Periodic spikes, all recover | Spike before each loss spike | Rejoins band |
| ~56,400 | max_grad_norm → 0.2 | Slightly tighter band | Softer spikes |
| ~72,233 | Update-norm clip ON | Further tightened | Still spiking |
| ~74k–80k | Update clip **OFF** (accident) | Explosions | **Worse plateau** |
| 80,000 | Team decision | — | Pivot to architecture |

---

### Lesson 9.4 — Phase 2: recovery without architecture (failed)

At **80k steps**, the team treated the run as **salvageable** — ~2.679T tokens of compute — and tried to recover **without** changing the attention stack.

#### Experiment: Necromancy (`exp1390_32b_necro`)

**Idea:** rebuild optimizer state and warm-start so update-norm statistics are sane again — fix "bad Adam moments" without throwing away weights.

**Outcome:** stabilized **briefly**, then **relapsed**.

**Lesson:** if the problem were optimizer state alone, necromancy would hold. Relapse → **attention** is the stress point.

#### Experiment: Muon (`exp1380_muon32b`)

**Idea:** swap optimizer (Muon), higher effective LR, different update geometry.

**Outcome:** abandoned when the run degraded again.

**Lesson:** temporary gradient health without fixing **attention at scale** is not enough.

```{mermaid}
flowchart LR
    P2[Phase 2 attempts] --> N[Necromancy]
    P2 --> M[Muon]
    N --> R1[Brief stability]
    M --> R2[Abandoned]
    R1 --> REL[Relapse]
    REL --> P3[Phase 3: QK-Norm]
```

Phase 2 tokens (~0.02T) are **diagnostic bursts** — excluded from the main cumulative story. They bought **evidence**, not a fix.

---

### Lesson 9.5 — Phase 3: QK-Norm warm-start (the fix)

This is the part people remember — and the graph-reading payoff.

#### What changed

| Component | Phase 1 (Llama 32B) | Phase 3 (Qwen3-style 32B) |
|-----------|---------------------|---------------------------|
| Attention | Standard Q/K projections | **QK-Norm** on Q and K per head |
| Embeddings | Trained 0→80k | **Preserved** from 80k checkpoint |
| MLP / FFN weights | Trained 0→80k | **Preserved** |
| Attention weights | Llama-style | **Re-learned** with normalized Q/K |
| Optimizer | AdamW (stressed state) | Re-warmup per report table |
| Hardware | v5p-512 | **v4-2048** (throughput/batch changes) |

**Code entrypoint:** `exp1395_qwen3_32b.py` (verify latest path on Marin `main`).

#### Warm-start mechanics (what transfers)

Think of the checkpoint as **layers of skill**:

```
Layer skill after 2.679T tokens:
  ✓ Token embeddings     — "what words mean" roughly
  ✓ FFN / MLP blocks     — factual and lexical knowledge
  ✗ Attention stack      — unstable dot-product geometry

Surgery:
  Keep embeddings + MLPs
  Replace attention module with Qwen3-style + QK-Norm
  Re-warm LR (~1,000 steps per retrospective table)
  Continue training
```

**They did not throw away 2.679T tokens of compute.** They changed the one subsystem that was structurally failing.

#### What to expect on W&B at the switch

1. **One-time loss penalty** — architecture mismatch at the seam
2. **~10B tokens** (retrospective) for train loss to recover to satisfaction
3. **Spikes stop** — grad/update norms in a tight band
4. **Throughput may step-change** — hardware transition v5p → v4

**ASCII — before and after:**

```
Before QK-Norm:    \__/\___/\____/\__   (spiky, exhausting)
                           ↑ architecture switch @ 80k
After QK-Norm:              \____________   (smooth, stable)
                             ~10B tokens to recover
```

#### Two "recovery clocks" (not a contradiction)

| Source | Timescale | What it measures |
|--------|-----------|------------------|
| **Retrospective (~10B tokens)** | Report-grade | Training loss recovery after switch — budget long-run stability |
| **Public thread (David Hall)** | Eyeballed plot / hundreds of steps | Faster "caught up on the curve" with particular smoothing |

Always ask on any plot: **raw or EMA?** **which loss?** **which token counter after restarts?**

---

### Lesson 9.6 — Phase 4: Mantis cooldown (data pipeline lessons)

Phase 4 (~1.074T tokens, steps 160k–192k) is a **data story** as much as a loss story.

#### Mix shift (Phase 1 → Phase 4)

| Category | Phase 1 (~%) | Phase 4 Mantis (~%) |
|----------|--------------|---------------------|
| Nemotron-CC (all tiers) | ~91 | ~68 |
| MegaMath (web, text, QA, code) | — | ~12 combined |
| arXiv, finemath, StackExchange, Wikipedia | small / absent | dedicated slices |

#### Bugs caught in cooldown (not "model can't do math")

| Issue | What happened | Fix |
|-------|---------------|-----|
| **GSM8k cache contamination** | Eval data leaked into training via caching | Replace with clean MegaMath |
| **LCG shuffle** | Poor cross-source randomization → batch skew | **Feistel shuffle** |

**Teaching point:** a benchmark jump after cooldown may reflect **data fixes**, not just "more training."

#### Benchmark results (Mantis)

| Benchmark | Marin 32B (Mantis) |
|-----------|-------------------|
| **Average** (suite) | **65.2%** |
| MMLU | 74.7% |
| BBH | 59.6% |
| HumanEval | 42.7% |
| GSM8K | 69.1% |
| vs OLMo 2 32B Base | **+2.0 avg**, better on 14/19 tasks |

The QK-Norm surgery produced a **competitive** model — Phase 1 tokens were not wasted.

---

### Lesson 9.7 — Visual storyboard (six panels)

**Panel 1 — Confident scale-up**

```
8B recipe (stable) ──────→ 32B scale-up (same recipe)
     ✓ worked                  ? will it hold?
```

A recipe stable at 8B can fail at 32B — stability headroom is not automatic.

**Panel 2 — Spikes appear but recover (0 → ~56k)**

Watch: grad_norm leads loss; val still trends down.

**Panel 3 — Mitigations soften but persist (~56k → ~74k)**

Grad clip → update clip → skip step. All **symptom-level**.

**Panel 4 — Bad plateau (~74k → 80k)**

Loss does not rejoin band. Eval shifts up. Norms stay high. **Structural.**

**Panel 5 — Architecture switch @ 80k**

Llama checkpoint → Qwen3-style + QK-Norm. Preserve embeddings + MLPs. Re-warm LR.

**Panel 6 — Boring is beautiful (Phase 3+)**

Smooth decay. Tight norm band. No emergency 3 AM hotfixes.

---

### FAQ: Marin 32B

#### Q: Was Marin "broken" for 2.679T tokens?

No — those tokens trained **embeddings and FFN knowledge** that transferred. The **attention geometry** was the unstable part; QK-Norm fixed the mechanism, warm-start preserved the rest.

#### Q: Why not start with QK-Norm from step 0?

Phase 1 followed a proven 8B recipe. QK-Norm adds complexity; teams often scale what worked small. Marin **discovered** the ceiling empirically — valuable public evidence.

#### Q: Could grad clipping alone have saved Phase 1?

No. Three progressively stronger clipping strategies **softened** spikes but spikes **persisted** until architecture changed.

#### Q: What should I log if I see Marin-like spikes?

Minimum set: **train loss (raw + smoothed)**, **val loss**, **grad_norm**, **update_norm**, **LR**, **tokens/s**, **max_grad_norm config**, **git SHA**, **data snapshot id**. Annotate W&B when you change any knob.

#### Q: How does this connect to Qwen3?

Qwen3 ships **QK-Norm from step 0** — see [Lesson 10](#lesson-10-qwen3-pretraining-stages). Marin is the **mid-run surgery** story; Qwen3 is the **design-in** story. Same stabilizer, different timeline.

### Check your understanding — Marin

- In one sentence: why did max_grad_norm 0.2 not fix the spikes?
- What three W&B signals distinguish "recoverable spike" from "bad plateau"?
- What weights were preserved in the Phase 3 warm-start? What was re-learned?
- Why is Phase 4 a data pipeline lesson, not an attention lesson?

---

## Lesson 10: Qwen3 pretraining stages

**Report:** [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388) (arXiv:2505.09388)

### Three stages (Section 3.2, paraphrased)

| Stage | Tokens / context | Data emphasis | Loss curve note |
|-------|------------------|---------------|-----------------|
| **S1 — General** | Tens of trillions, 4,096 ctx | Broad web | Classic steep drop → slow decline |
| **S2 — Reasoning** | ~5T more, 4,096 ctx | Higher STEM/code/reasoning/synthetic | **Accelerated LR decay** — slope change at boundary is intentional |
| **S3 — Long context** | Hundreds of billions, 32,768 ctx | Long-doc corpora; RoPE/YARN/DCA | Do not compare S3 loss to S1 without context-length accounting |

**QK-Norm is present from step 0** — unlike Marin, no mid-run surgery. Stability is baked in.

### How to read Qwen3 figures

1. Open PDF, search "pre-training" or "loss"
2. Read **caption first** — x-axis per-stage or cumulative?
3. Look for **stage boundary markers**
4. Ask: raw or smoothed? train or eval?

---

## Lesson 11: Boring is beautiful

After Marin, training looks like drama — spikes, rescues, surgery. The real lesson: **the goal of all that work is a boring loss curve.**

| Boring (good) | Exciting (bad) |
|---------------|----------------|
| Smooth monotonic decline | Spikes, plateaus, recoveries |
| Grad norms in tight band | Norm explosions |
| Throughput flat | Throughput drops |
| Eval tracks train down | Eval diverges |
| No mid-run code changes | Emergency hotfixes at 3 AM |

**"Boring" does not mean "easy."** Marin took months and multiple failed interventions to reach boring. QK-Norm gives **stability headroom** so future runs start boring.

### QK-Norm in the wild (snapshot — verify on current cards)

| Model / line | QK-Norm? | Notes |
|--------------|----------|-------|
| **Qwen3** | Yes | GQA, SwiGLU — arXiv:2505.09388 |
| **OLMo 3** | Yes | AI2 release |
| **SmolLM3** | Yes | Smol training playbook |
| **Sarvam** | Yes | 30B GQA; 105B MLA at scale |
| **MiniMax** | Yes | Production-stability focus |
| **Kimi K2** | No (QK-clip-style) | MLA efficiency — different toolkit |

Do not overfit "add QK-Norm" to every stack — **Kimi-style** training uses different stabilizers.

---

## Takeaways and exercises

### Takeaways

- Fix **batch/token axes** before comparing runs; fix **pipeline** before panic-reading loss
- Use **val + train + norms + LR + throughput** as a set
- **Data mix is the first decision** — read the mix table before interpreting loss
- **Spikes are diagnostic** — Marin shows clipping necessary but insufficient when root is attention
- **"Boring is beautiful"** — stability engineering targets a featureless curve
- **Cite primary sources** — Marin retrospective, OLMo paper, Qwen3 report

### Exercises

1. **OLMo checkpoints** — Enumerate 10 `revision` tags, pick two far apart, run three prompts, tabulate differences
2. **OLMo weights** — Relative embedding change early vs late; 3-sentence interpretation
3. **W&B** — Add YAML config snapshot to next run; compare SHA + data path when weird
4. **Marin timeline** — Read Phase 1–3 with Data Mix + Optimizer pages; one-page timeline in your words
5. **Data mix** — Marin table vs OLMo Dolma: 5-row comparison with interpretation
6. **Qwen3** — Find three pretrain stages in arXiv:2505.09388; note x-axis on loss figures
7. **Perfetto** — One trace, three bullet findings (data vs comm vs kernel)
8. **Marin W&B mock** — Sketch six panels (loss, val, grad, update, LR, tok/s) for steps 56k, 74k, 80k, 90k; label what you'd expect in each

---

## References

**Marin**

- [Marin 32B Retrospective](https://marin-community.github.io/marin/retrospective/32b/) (primary)
- [marin-community/marin-32b-base](https://huggingface.co/marin-community/marin-32b-base)
- exp1295_32b (Phase 1), exp1395 Qwen3 32B (Phase 3) — Marin GitHub

**Qwen3**

- [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388)
- [Qwen3 inference guide (this repo)](qwen3.c.md)

**OLMo / Dolma**

- [OLMo paper](https://arxiv.org/abs/2402.00838)
- [OLMo 2](https://arxiv.org/abs/2501.00656)
- [allenai/OLMo-7B-0424-hf](https://huggingface.co/allenai/OLMo-7B-0424-hf)
- [Dolma dataset](https://huggingface.co/datasets/allenai/dolma)

**Tools**

- [Weights & Biases](https://wandb.ai/)
- [Perfetto UI](https://ui.perfetto.dev/)
