---
title: "RL-Like Fork Sampling — Viz Branches and Training Roadmap"
description: "How dual-generation branching in the Qwen3 sampling visualizer connects to RL rollouts, KV snapshots vs re-prefill, and what HF TRL / veRL / prime-rl actually do. A First Break AI roadmap for learning and building."
date: 2026-06-05
categories: [rl, grpo, sampling, kv-cache, visualization, qwen3, first-break-ai]
---

> **First Break AI — Step 2 → Step 3 bridge**
>
> This post documents two related tracks:
> 1. **Viz fork (for learning)** — interactive dual-generation in the sampling dashboard
> 2. **RL training (for doing)** — GRPO-style rollouts with visualization at every step
>
> Part of [First Break AI](https://cohort.bubblnet.com/).

---

# RL-Like Checkpoint Sampling: Viz Forks and the Training Roadmap

You already have a live sampling visualizer: temperature, top-p, nucleus stats, replay by clicking tokens. The next step is **dual generation** — see what the model chose *and* what would have happened if you picked a different token.

This post explains how that fork works in `qwen3.c`, how it relates to RL (and how it differs from what TRL / veRL / prime-rl do), and lists concrete **things to do** for both tracks.

---

## The Big Picture

```{mermaid}
flowchart TD
    subgraph today [What exists today]
        runviz["run_viz emits sample_step events"]
        bridge["sampling-bridge.mjs"]
        dash["Next.js dashboard: live + replay"]
        runviz --> bridge --> dash
    end

    subgraph fork [Track 1: Viz fork for learning]
        click["User clicks alt token at step N"]
        snapshot["KV snapshot at fork OR re-prefill"]
        branch["Force token N, continue decode"]
        compare["Side-by-side: default vs branch"]
        click --> snapshot --> branch --> compare
    end

    subgraph rl [Track 2: RL with visualization]
        gro["G rollouts per prompt"]
        reward["Score completions"]
        vizrl["Viz every rollout + advantage heatmap"]
        update["Policy update via GRPO"]
        gro --> reward --> vizrl --> update
    end

    dash --> fork
    fork -.->|"same MDP mental model"| rl
```

---

## Part 1: Viz Fork — Dual Generation for Learning

### What you have now

The current pipeline is **observational**:

| Layer | File | Role |
|-------|------|------|
| C engine | `repos/qwen3.c/run.c` | `emit_sample_event()` on decode steps; `-v 1` |
| Bridge | `tools/sampling-bridge.mjs` | Parses stderr, WebSocket broadcast, saves `sampling-session.json` |
| Dashboard | `apps/sampling-viz/` | Live bars, nucleus stats, clickable timeline replay |

Clicking a timeline token **replays** that step's probabilities. It does **not** change the model's future.

### What dual generation adds

**Example:** The model outputs:

> The book "The Jungle Book" by Rudyard Kipling does not have a king in the traditional sense. However, it is a fictional jungle kingdom in the context of the **story**...

At step 400, top candidates might be `"story"` (chosen), `"movie"`, `"book"`. You click **"movie"** and see a **branch**:

> ...in the context of the **movie**. The 1967 Disney adaptation portrays...

Same prefix up to 399. Different token at 400. Different continuation from 401 onward.

```{mermaid}
flowchart TD
    prefix["Shared prefix: positions 0..399"]
    fork["Fork at position 400"]
    default["Default: sample() chose story"]
    branch["Branch: user forced movie"]
    suffix1["Suffix KV tree 1: 401..end"]
    suffix2["Suffix KV tree 2: 401..end"]

    prefix --> fork
    fork --> default --> suffix1
    fork --> branch --> suffix2
```

This is **Pattern B** (mid-trajectory fork). It is ideal for teaching nucleus sampling and counterfactual reasoning.

---

### Two implementation options (detailed)

Both reach the same logits at the fork. They differ in speed and memory.

#### Option A: KV snapshot (recommended for viz)

**How it works**

At fork position `N`, copy `key_cache` and `value_cache` up to `N-1` from `RunState`:

```c
// run.c — RunState already has:
float* key_cache;   // (layer, seq_len, kv_dim)
float* value_cache; // (layer, seq_len, kv_dim)
```

Snapshot size for prefix up to position 400 (Qwen3 0.6B):

```
bytes = 2 × n_layers × N × kv_dim × sizeof(float)
      = 2 × 28 × 400 × 1024 × 4
      ≈ 91 MB
```

Then for each branch:

1. Restore snapshot into a branch `RunState`
2. **Force** token at position `N` (skip `sample()`)
3. `forward(forced_token, N)` — writes K,V at pos N into branch cache
4. Continue decode with normal sampling from `N+1`

**Pros**

- Fork is ~instant after snapshot exists
- Best UX for interactive "click and explore"
- Matches how tree search / MCTS thinks about state

**Cons**

- ~91 MB per snapshot at pos 400 (grows with N)
- Naive "10 branches = 10 full copies" is wasteful (~910 MB)
- Smart version: **1 read-only prefix** + **10 independent suffix trees** (only pos N..end differ)

| Piece | Count | Memory role |
|-------|-------|-------------|
| Prefix KV (0..N-1) | 1 shared | Read-only snapshot |
| Suffix KV (N..end) | 1 per branch | Grows independently per try |

#### Option B: Re-prefill (no snapshot)

**How it works**

1. Reset `RunState` (zero KV cache)
2. Re-run `forward()` for every token in positions `0..N-1` from saved token list
3. Force token at position `N`
4. Continue decode

**Pros**

- No snapshot memory
- Simpler to implement first
- Same math as snapshot

**Cons**

- Slow: ~1–3 seconds to replay 400 tokens on CPU
- 10 branches = 10 full re-prefills (unless you cache prefix once and only branch at N)

| Approach | How | Pros | Cons |
|----------|-----|------|------|
| **KV snapshot** | Copy `key_cache`/`value_cache` at fork, restore, force token, continue | Fast fork (~instant) | ~91 MB per prefix at pos 400; 10 suffix trees during active branches |
| **Re-prefill** | Reset state, re-run tokens `0..N-1`, force token `N`, continue | No snapshot RAM | Slower (~1–3 s for pos 400); 10 branches = 10× prefill cost |

**Recommendation:** Start with **re-prefill** for Phase 1 (prove correctness). Move to **shared-prefix KV snapshot** for Phase 2 (interactive UX).

---

### Architecture changes (viz fork)

```{mermaid}
sequenceDiagram
    participant UI as Dashboard
    participant Bridge as sampling-bridge
    participant C as run_viz

    C->>Bridge: sample_step pos=400 chosen=story
    Bridge->>UI: broadcast event
    UI->>Bridge: fork pos=400 token_id=movie
    Bridge->>C: FORK command via control channel
    C->>C: restore KV or re-prefill to 399
    C->>C: force movie at 400, continue decode
    C->>Bridge: sample_step path=branch
    Bridge->>UI: branch timeline + events
```

**Files to touch**

| File | Changes |
|------|---------|
| `repos/qwen3.c/run.c` | `save_checkpoint()`, `restore_checkpoint()`, `decode_with_forced_token()`, fork control on stdin or side channel |
| `tools/sampling-bridge.mjs` | Bidirectional WebSocket: `fork` request in, `path: default\|branch` tag on events |
| `apps/sampling-viz/app/page.tsx` | Clickable top-20 bars, split-pane default vs branch, fork banner |
| `apps/sampling-viz/app/globals.css` | Branch path styling (e.g. blue vs gold default) |

---

### Viz fork — things to do (checklist)

**Phase 1: Fork from replay (offline, re-prefill)**

- [ ] Save full token sequence in `sampling-session.json` (prefix tokens per step, not just chosen_piece)
- [ ] Add `fork` command to bridge: `{ type: "fork", pos, token_id }`
- [ ] C: `fork_from_session(tokens[], pos, forced_id)` — reset, re-prefill, force, decode N more tokens
- [ ] Dashboard: "Explore this path" button on each top-20 bar row
- [ ] Split pane: left = default path text, right = branch path text
- [ ] Highlight divergence point in both timelines

**Phase 2: KV snapshot (fast interactive fork)**

- [ ] C: `Checkpoint` struct — copy `key_cache`/`value_cache` up to `pos`, save token buffer + pos
- [ ] Shared read-only prefix + per-branch suffix allocation
- [ ] Snapshot on each decode step when `-v 1` (optional ring buffer of last K checkpoints)
- [ ] Fork during live session without stopping default path

**Phase 3: Multi-branch tree (10 tries at fork)**

- [ ] User picks top-K tokens at one step → K parallel branches
- [ ] Tree UI: shared trunk, fan-out at fork, click any leaf
- [ ] Compare rewards manually ("which continuation is better?") as RL intuition builder

**Phase 4: Pedagogy hooks**

- [ ] Callout panel: "This is what GRPO does automatically with G random rollouts"
- [ ] Side-by-side entropy / nucleus mass at fork for default vs branch first step
- [ ] Export fork session JSON for classroom demos

---

## Part 2: RL Training — GRPO and Visualization

### How standard RL libs do it (confirmed)

**HuggingFace TRL**, **veRL**, and **Prime Intellect prime-rl** use **Pattern A** — not mid-answer forks.

| Library | Parameter | Meaning |
|---------|-----------|---------|
| TRL | `num_generations=G` | G full completions per prompt |
| veRL | `rollout.n` + prefix caching | G sequences; shared **prompt** KV via vLLM |
| prime-rl | `rollouts_per_example=G` | G trajectories per prompt for advantage |

```{mermaid}
flowchart LR
    prompt["Prompt tokens 0..P"]
    fork["Fork at end of prompt"]
    r1["Rollout 1: full completion"]
    r2["Rollout 2: full completion"]
    rG["Rollout G: full completion"]

    prompt --> fork
    fork --> r1
    fork --> r2
    fork --> rG
```

**What happens to KV**

| When | KV behavior |
|------|-------------|
| During each rollout | Live KV cache built incrementally (inside vLLM / `generate()`) |
| Across G rollouts | Prompt prefix often **cached** (`enable_prefix_caching` in veRL) |
| After rollout | KV **discarded**; keep `prompt_ids`, `completion_ids`, `logprobs`, `reward` |
| On weight update | `clear_kv_cache()` — stale policy KV must not persist |

**What they do NOT do (standard GRPO)**

- Fork at step 400 inside the answer
- Store 10 KV suffix trees from one mid-trajectory fork
- User-forced counterfactual tokens

They don't need Pattern B because:

1. **Diversity** — temperature + G full rollouts diverge naturally (often at token 1 of the answer)
2. **Training signal** — "which full completion was best?" not "what if token X at step 400?"
3. **Batching** — G equal-shaped rollouts batch cleanly on GPU; irregular trees do not
4. **Prefix caching** — already dedupes the expensive shared part (the prompt)

### The MDP connection (why viz fork still teaches RL)

Both are the same Markov decision process:

| RL term | Viz term |
|---------|----------|
| State `s` | Token prefix (or KV cache encoding it) |
| Action `a` | Next token ID |
| Policy `π(a\|s)` | Softmax after temperature + top-p |
| Trajectory | Full generated text |
| Reward | Human judgment or automated scorer |

**GRPO:** sample G actions stochastically from the root → compare G trajectories → reinforce better ones.

**Viz fork:** pick one action manually at step N → compare 2 trajectories → *see* why sampling matters.

---

### RL with heavy visualization — things to do (checklist)

**Phase 1: Multi-rollout viz (Pattern A, no training yet)**

- [ ] Script: run same prompt G times with different `-s <seed>` values
- [ ] Dashboard: "compare rollouts" view — G columns, same prompt, different completions
- [ ] Show where rollouts **first diverge** (highlight earliest differing token)
- [ ] Manual reward: user clicks "best completion" → store preference pair (DPO intuition)

**Phase 2: Reward + advantage display (still inference-only)**

- [ ] Plug in simple reward: exact match, regex, or length penalty
- [ ] Compute group mean/std across G rewards (GRPO advantage formula)
- [ ] Heatmap: per-token logprob colored by advantage of its rollout
- [ ] Panel: "Rollout 3 won — these tokens had high logprob"

**Phase 3: Minimal GRPO loop on Qwen3 0.6B**

- [ ] Export logits + logprobs from `run.c` for training steps (or wrap with TRL on HF checkpoint)
- [ ] Use TRL `GRPOTrainer` with `num_generations=4` on a tiny prompt dataset
- [ ] **Viz bridge extended:** stream all G rollouts live to dashboard during training
- [ ] Training dashboard: loss, mean reward, KL, per-step rollout gallery

**Phase 4: Full training stack (production path)**

- [ ] veRL or prime-rl for scaled rollouts; keep custom viz as parallel WebSocket tap
- [ ] Prefix caching observability: log cache hit rate per prompt
- [ ] Document: "our viz fork = Pattern B; TRL = Pattern A" with live demo

**Phase 5: Advanced (optional)**

- [ ] Process reward model on fork steps (score "was movie a good choice at 400?")
- [ ] Tree search viz: expand top-K at each step, prune by reward
- [ ] Connect to First Break AI Step 3 / Step 4 curriculum

---

## Side-by-side summary

| | **Viz fork (Pattern B)** | **GRPO in TRL/veRL/prime-rl (Pattern A)** |
|---|---|---|
| **Goal** | Learn sampling; explore counterfactuals | Train model weights |
| **Fork point** | Any step user clicks | End of prompt only |
| **Branches** | User-chosen token(s) | Random sampling |
| **KV strategy** | Snapshot or re-prefill at fork | Live cache; prompt prefix cached; discard after |
| **Stored long-term** | Session JSON + token paths | Tokens + logprobs + rewards |
| **G tries** | K branches from one fork | G full rollouts from prompt |
| **Libraries** | Custom `qwen3.c` + dashboard | TRL, veRL, prime-rl |

---

## Suggested build order

1. **Now:** Use existing viz for temperature / top-p experiments (`-t 0.2` vs `-t 1.2`, `-k 1` thinking mode)
2. **Next:** Phase 1 viz fork (re-prefill, offline from saved session)
3. **Then:** KV snapshot for fast live fork
4. **Then:** Multi-rollout compare view (Pattern A viz — mirrors TRL)
5. **Later:** TRL GRPO on small dataset with training viz bridge

---

## References

- [First Break AI cohort](https://cohort.bubblnet.com/)
- [Step 2: Run a model locally](qwen3.c.md)
- [HuggingFace TRL GRPO](https://huggingface.co/docs/course/main/en/chapter12/4)
- [veRL rollout KV offload](https://verl.readthedocs.io/en/latest/perf/rollout_kv_offload.html)
- [Prime Intellect prime-rl training](https://primeintellect-ai-verifiers.mintlify.app/guides/training)
- [DeepSeek-R1 — 64 samples per query for pass@1](https://huggingface.co/deepseek-ai/DeepSeek-R1)
- [Qwen3 sampling configs (thinking vs non-thinking)](https://huggingface.co/Qwen/Qwen3-0.6B)

---

Part of [First Break AI](https://cohort.bubblnet.com/) | [Discord](https://discord.gg/hRPese4H3F) | [GitHub](https://github.com/thefirehacker/firstbreakai)
