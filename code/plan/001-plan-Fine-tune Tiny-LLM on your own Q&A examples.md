# Fine-tune Tiny-LLM on your own Q&A examples

## Context

You chatted with the local `arnir0/Tiny-LLM` model via `chat.py` and got incoherent rambling (e.g. asked how to download DeepSeek-V3 from Hugging Face, got nonsense). You want to "update" the model based on chats — i.e. run a small fine-tuning pass so it visibly shifts its behavior toward examples you feed it.

**Reality check, stated plainly so expectations are calibrated:** Tiny-LLM is a genuine ~10M-parameter, 1-layer base language model pretrained on FineWeb (per its own model card) — it was never instruction-tuned. Fine-tuning it on a handful of examples will not turn it into a capable assistant. The goal of this experiment is to demonstrate the fine-tuning loop and see the model's output visibly overfit toward your example answers (e.g. it starts mimicking the *style* and fragments of the DeepSeek-download answer when asked something similar) — not to produce a good chatbot.

There's no existing training code anywhere in this repo to build on (confirmed via exploration) — this is new, self-contained scaffolding, kept dependency-light to match how `chat.py`/`run_inference.py` already work (plain `torch` + `transformers`, no new heavy deps).

## Environment note — check first

Running `python3 -c "import transformers"` in this tool's shell currently fails:
```
ImportError: huggingface-hub>=0.34.0,<1.0 is required for a normal functioning of this module, but found huggingface-hub==1.14.0.
```
However, your own terminal successfully ran `chat.py` earlier in this session, so your interactive shell may have a different environment already working. **Before running `finetune.py`, first confirm `python3 -c "import transformers"` works in your actual terminal.** If it fails there too, fix with:
```bash
pip install "huggingface_hub<1.0,>=0.34.0"
```
This is a version pin, not a new dependency.

## Files to add/edit (all under `models/Tiny-LLM/`)

### 1. `train_data.jsonl` (new) — your editable training examples
One JSON object per line: `{"prompt": ..., "response": ...}`. Plain-text, diff-friendly, easy for you to hand-append more examples over time as you find bad answers you want to correct. Seed with ~8-10 short, consistent-style pairs, including a good version of the DeepSeek question, e.g.:
```jsonl
{"prompt": "How do I download the latest DeepSeek model from Hugging Face?", "response": "Install huggingface_hub, then run `huggingface-cli download deepseek-ai/DeepSeek-V3 --local-dir ./DeepSeek-V3`. You can also use snapshot_download() in Python."}
{"prompt": "What is Python used for?", "response": "Python is a general-purpose programming language used for web development, data science, automation, and machine learning."}
```
plus several more similar factual Q&A pairs, kept short and uniform in tone so the tiny model has a repeatable pattern to latch onto.

### 2. `finetune.py` (new) — the training script

- Loads base model/tokenizer from `MODEL_PATH = "."` (same convention as `chat.py`).
- Sets `tokenizer.pad_token = tokenizer.eos_token` (tokenizer has no pad token — confirmed in `tokenizer_config.json`) and `model.config.pad_token_id` accordingly.
- Formats each example as `"Question: {prompt}\nAnswer: {response}"` + explicit EOS (the tokenizer auto-adds BOS but not EOS, per `add_bos_token: true` / `add_eos_token: false` — so appending EOS manually is what teaches the model to stop instead of rambling).
- Custom `torch.utils.data.Dataset`: tokenizes the prompt portion and the full text *separately* (not by slicing token ids) to get a clean boundary, then builds `labels` with `-100` on every prompt token so loss is computed only on the response — standard SFT masking.
- Custom collate function pads `input_ids`/`attention_mask` normally but pads `labels` with `-100` (not the pad token id) — critical because `pad_token_id == eos_token_id` here, so padding must never overwrite the one legitimate EOS label with masking-by-id.
- **Plain PyTorch training loop, no `transformers.Trainer`.** `Trainer` in the installed `transformers` version hard-requires `accelerate`, which isn't installed and buys nothing for a single-process, ~10M-param, CPU/MPS job. A ~30-line manual loop (`AdamW`, gradient clipping, `model(**batch).loss` which HF's `LlamaForCausalLM` already shifts/masks internally) is simpler and more transparent.
- Defaults: `--epochs 40`, `--lr 5e-4`, `--batch-size min(4, len(dataset))`, grad-norm clip 1.0, `max_length=256`. No scheduler/warmup — kept simple for a demo this size. All overridable via argparse.
- Built-in before/after smoke test: generate on a couple of sanity prompts (including the DeepSeek one, verbatim from the dataset) using the model *before* training, print as `BEFORE:`, then again *after* training as `AFTER:` — so one script run directly shows the shift you're after.
- Saves to `models/Tiny-LLM/finetuned/` via `model.save_pretrained()` / `tokenizer.save_pretrained()`, leaving the original base weights in `models/Tiny-LLM/` untouched. Also writes `finetuned/chat_format.json` with the prompt template, so `chat.py` can auto-detect and reuse it later. `models/*` is already gitignored, so this new checkpoint directory won't be committed.

### 3. `chat.py` (edit) — pick which checkpoint to talk to

- Add an `argparse` `--model-path` flag defaulting to `"."`, replacing the hardcoded `MODEL_PATH = "."`. Usage stays `python chat.py` for the base model, or `python chat.py --model-path finetuned` for the fine-tuned one.
- On startup, check for `<model_path>/chat_format.json`; if present, wrap each user input through its saved template before tokenizing (and strip the wrapped text's length off the decoded output, not the raw input's length). If absent (base model case), behavior is unchanged from today.

## Verification

1. From `models/Tiny-LLM/`, confirm `python3 -c "import transformers"` works in your terminal (fix per the environment note above if not).
2. Run `python3 finetune.py` — watch the per-epoch loss print and decrease, then compare the printed `BEFORE:`/`AFTER:` generations for the seeded sanity prompts.
3. Run `python3 chat.py --model-path finetuned` and ask the same DeepSeek-download question interactively — confirm the answer visibly shifts toward the trained example (won't be perfect, but should show clear influence).
4. Run `python3 chat.py` (no flag) and confirm the base model's behavior is unchanged from before.
5. To iterate: append more `{"prompt":..., "response":...}` lines to `train_data.jsonl` and rerun `finetune.py`.
