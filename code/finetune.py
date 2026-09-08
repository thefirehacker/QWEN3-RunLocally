import argparse
import json
import os
from datetime import datetime

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

TEMPLATE = "Question: {prompt}\nAnswer:"

SANITY_PROMPTS = [
    "How do I download the latest DeepSeek model from Hugging Face?",
    "how to download qwen3 then ?",
    "what is architecture of deepseek ?",
    "are there different kinds of models?",
    "What is Python used for?",
]


class SFTDataset(Dataset):
    def __init__(self, path, tokenizer, max_length=384):
        self.tokenizer = tokenizer
        self.max_length = max_length
        with open(path) as f:
            self.rows = [json.loads(line) for line in f if line.strip()]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        prompt_text = TEMPLATE.format(prompt=row["prompt"])
        full_text = prompt_text + " " + row["response"]

        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=True)["input_ids"]
        full_ids = self.tokenizer(full_text, add_special_tokens=True)["input_ids"]
        full_ids = (full_ids + [self.tokenizer.eos_token_id])[: self.max_length]

        split = min(len(prompt_ids), len(full_ids))
        labels = list(full_ids)
        labels[:split] = [-100] * split

        return {
            "input_ids": full_ids,
            "labels": labels,
            "attention_mask": [1] * len(full_ids),
        }


def collate(batch, pad_id):
    max_len = max(len(ex["input_ids"]) for ex in batch)
    input_ids, attn, labels = [], [], []
    for ex in batch:
        pad = max_len - len(ex["input_ids"])
        input_ids.append(ex["input_ids"] + [pad_id] * pad)
        attn.append(ex["attention_mask"] + [0] * pad)
        labels.append(ex["labels"] + [-100] * pad)
    return {
        "input_ids": torch.tensor(input_ids),
        "attention_mask": torch.tensor(attn),
        "labels": torch.tensor(labels),
    }


def generate(model, tokenizer, prompt, device, max_new_tokens=128):
    text = TEMPLATE.format(prompt=prompt)
    inputs = tokenizer(text, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=1.0,
        top_k=50,
        top_p=0.95,
        pad_token_id=tokenizer.eos_token_id,
    )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded[len(text):].strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="../models/Tiny-LLM")
    parser.add_argument(
        "--data",
        default=os.path.join(os.path.dirname(__file__), "tool-call", "reasoning.jsonl"),
    )
    parser.add_argument("--output-root", default="../models/Tiny-LLM/finetuned",
                         help="Parent folder each run's timestamped checkpoint is saved under")
    parser.add_argument("--output-dir", default=None,
                         help="Exact checkpoint folder to save to; overrides --output-root/timestamp naming")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(args.output_root, datetime.now().strftime("%Y%m%d-%H%M%S"))

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    device = "cpu"  # MPS doesn't support the int64 cumsum op transformers' Llama generation code uses
    model.to(device)

    print("=== BEFORE fine-tuning ===")
    model.eval()
    for p in SANITY_PROMPTS:
        print(f"Q: {p}\nA: {generate(model, tokenizer, p, device, args.max_new_tokens)}\n")

    dataset = SFTDataset(args.data, tokenizer, max_length=args.max_length)
    batch_size = min(args.batch_size, len(dataset))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate(b, tokenizer.pad_token_id),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    model.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
        print(f"epoch {epoch + 1}/{args.epochs}  loss={total_loss / len(loader):.4f}")

    print("\n=== AFTER fine-tuning ===")
    model.eval()
    for p in SANITY_PROMPTS:
        print(f"Q: {p}\nA: {generate(model, tokenizer, p, device, args.max_new_tokens)}\n")

    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    with open(os.path.join(output_dir, "chat_format.json"), "w") as f:
        json.dump({"template": TEMPLATE}, f, indent=2)
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump({
            "base_model_path": args.model_path,
            "data": args.data,
            "epochs": args.epochs,
            "lr": args.lr,
            "trained_at": datetime.now().isoformat(timespec="seconds"),
        }, f, indent=2)

    print(f"Saved fine-tuned model to {output_dir}")
    print(f"Continue training from this checkpoint with: --model-path {output_dir}")


if __name__ == "__main__":
    main()
