# Tiny-LLM scripts

Scripts for running and fine-tuning the local `arnir0/Tiny-LLM` model. The model
weights themselves live in `../models/Tiny-LLM/` (not in this folder) — these
scripts just point at that path.

## Setup

```bash
pip install torch transformers
```

If `import transformers` fails with a `huggingface-hub` version error, pin it:
```bash
pip install "huggingface_hub<1.0,>=0.34.0"
```

## Files

| File | Purpose |
|---|---|
| `run_inference.py` | One-shot generation on a hardcoded prompt, from the base model |
| `chat.py` | Interactive chat loop; works with the base model or a fine-tuned checkpoint |
| `train_data.jsonl` | Original short Q&A pairs (easy to overfit / ignore the question) |
| `tool-call/reasoning.jsonl` | Chain-of-thought pairs: restate the question, then answer |
| `finetune.py` | Fine-tunes a base model or checkpoint (defaults to `tool-call/reasoning.jsonl`), saving a new timestamped checkpoint each run |
| `Inferencecode.py` | Original model-card example; downloads from the Hugging Face Hub instead of using local files |

## Running the base model

```bash
python3 run_inference.py
# or, interactively:
python3 chat.py
```
Both default to `../models/Tiny-LLM`.

## Fine-tuning

1. Edit `tool-call/reasoning.jsonl` — each line is
   `{"prompt": ..., "response": "Reason: ...\\nAnswer: ..."}`.
   The `Reason:` line names the actual subject so the model practices
   reading the question instead of copying one memorized reply.
2. Run:
   ```bash
   python3 finetune.py
   ```
   This prints a `BEFORE:` generation, trains for 20 epochs (loss should trend
   down each epoch), then prints an `AFTER:` generation so you can compare.
   To train on the old short answers instead:
   `python3 finetune.py --data train_data.jsonl --epochs 40`

**Where the fine-tuned checkpoint is saved:**
Each run creates a new, timestamped folder so runs never overwrite each other:
```
../models/Tiny-LLM/finetuned/<YYYYMMDD-HHMMSS>/
```
e.g. `/Users/booimac/AIEDX/Code/AI/Qwen3-RunLocally/models/Tiny-LLM/finetuned/20260908-193045/`.
`finetune.py` prints the exact path at the end of the run. Each checkpoint
folder also has a `metadata.json` recording which base model and dataset
produced it, and a `chat_format.json` used by `chat.py`.

By default `--model-path` (the model being trained) is the original base
model in `../models/Tiny-LLM/`, which is never touched by training — pass a
different `--model-path`/`--data` to change what's fine-tuned and on what.

## Incremental training (continue from a previous checkpoint)

Instead of always training from the base model, you can pick up any previous
checkpoint and keep training it further — pass its folder as `--model-path`:
```bash
python3 finetune.py \
  --model-path ../models/Tiny-LLM/finetuned/20260908-193045 \
  --data tool-call/reasoning.jsonl
```
This still writes a *new* timestamped folder under `finetuned/` (it never
overwrites the checkpoint you started from), so you build up a chain of
checkpoints over time, each traceable via its `metadata.json`.

Full list of args you can mix and match:
```bash
python3 finetune.py \
  --model-path <base model or checkpoint to train from> \
  --data <path to a .jsonl dataset> \
  --output-root <parent folder for the new timestamped checkpoint> \
  --output-dir <exact folder, if you don't want a timestamp> \
  --epochs <int> --lr <float> --batch-size <int>
```

## Chatting with a fine-tuned checkpoint

```bash
python3 chat.py --model-path ../models/Tiny-LLM/finetuned/20260908-193045
```
`chat.py` auto-detects that checkpoint's `chat_format.json` and wraps your
input in the same `Question: {prompt}\nAnswer:` template it was trained on —
no extra flags needed for that part.

To compare against the untouched base model:
```bash
python3 chat.py --model-path ../models/Tiny-LLM
```

## Notes

- Tiny-LLM is a ~10M-parameter, 1-layer base model — CoT fine-tuning teaches a
  *format* (restate the subject, then answer), not real reasoning. If it still
  copies one reply for every question, add more contrastive pairs or lower
  `--epochs`.
- `finetune.py` always runs on CPU. MPS (Apple GPU) is skipped because
  `transformers`' Llama generation code uses an int64 `cumsum` op MPS doesn't
  support — CPU is already fast enough at this model size regardless.
