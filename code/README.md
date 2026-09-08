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
| `train_data.jsonl` | Your editable Q&A training examples (`{"prompt": ..., "response": ...}` per line) |
| `finetune.py` | Fine-tunes the base model on `train_data.jsonl` |
| `Inferencecode.py` | Original model-card example; downloads from the Hugging Face Hub instead of using local files |

## Running the base model

```bash
python3 run_inference.py
# or, interactively:
python3 chat.py
```
Both default to `../models/Tiny-LLM`.

## Fine-tuning

1. Edit `train_data.jsonl` — add/change `{"prompt": ..., "response": ...}` lines.
2. Run:
   ```bash
   python3 finetune.py
   ```
   This prints a `BEFORE:` generation, trains for 40 epochs (loss should trend
   down each epoch), then prints an `AFTER:` generation so you can compare.

**Where the fine-tuned checkpoint is saved:**
```
../models/Tiny-LLM/finetuned/
```
(i.e. `/Users/booimac/AIEDX/Code/AI/Qwen3-RunLocally/models/Tiny-LLM/finetuned`)

The original base weights in `../models/Tiny-LLM/` are never touched — every
run of `finetune.py` retrains from the original base, so runs don't compound.

## Chatting with the fine-tuned checkpoint

```bash
python3 chat.py --model-path ../models/Tiny-LLM/finetuned
```
`chat.py` auto-detects `finetuned/chat_format.json` and wraps your input in
the same `Question: {prompt}\nAnswer:` template the model was trained on —
no extra flags needed for that part.

To compare against the untouched base model:
```bash
python3 chat.py --model-path ../models/Tiny-LLM
```

## Notes

- Tiny-LLM is a ~10M-parameter, 1-layer base model — fine-tuning on a handful
  of examples will make it visibly mimic those examples, not turn it into a
  capable assistant. With very few/unbalanced examples it can also collapse
  onto one dominant answer regardless of the question — add more varied
  examples or lower `--epochs` if that happens.
- `finetune.py` always runs on CPU. MPS (Apple GPU) is skipped because
  `transformers`' Llama generation code uses an int64 `cumsum` op MPS doesn't
  support — CPU is already fast enough at this model size regardless.
