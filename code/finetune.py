import argparse
import json
import os

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

TEMPLATE = "Question: {prompt}\nAnswer:"

SANITY_PROMPTS = [
    "How do I download the latest DeepSeek model from Hugging Face?",
    "What is Python used for?",
]


class SFTDataset(Dataset):
    def __init__(self, path, tokenizer, max_length=256):
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


def generate(model, tokenizer, prompt, device, max_new_tokens=60):
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
    parser.add_argument("--data", default=os.path.join(os.path.dirname(__file__), "train_data.jsonl"))
    parser.add_argument("--output-dir", default="../models/Tiny-LLM/finetuned")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device)

    print("=== BEFORE fine-tuning ===")
    model.eval()
    for p in SANITY_PROMPTS:
        print(f"Q: {p}\nA: {generate(model, tokenizer, p, device)}\n")

    dataset = SFTDataset(args.data, tokenizer)
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
        print(f"Q: {p}\nA: {generate(model, tokenizer, p, device)}\n")

    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    with open(os.path.join(args.output_dir, "chat_format.json"), "w") as f:
        json.dump({"template": TEMPLATE}, f, indent=2)

    print(f"Saved fine-tuned model to {args.output_dir}")


if __name__ == "__main__":
    main()
