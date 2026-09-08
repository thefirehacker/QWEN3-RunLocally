import argparse
import json
import os

from transformers import AutoModelForCausalLM, AutoTokenizer


def load_chat_format(model_path):
    path = os.path.join(model_path, "chat_format.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)["template"]
    return None


def generate_text(prompt, model, tokenizer, template, max_new_tokens=100, temperature=1.0, top_k=50, top_p=0.95):
    text = template.format(prompt=prompt) if template else prompt
    inputs = tokenizer(text, return_tensors="pt")

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded[len(text):].strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="../models/Tiny-LLM",
                         help="Base model folder, or a fine-tuned checkpoint (e.g. ../models/Tiny-LLM/finetuned)")
    args = parser.parse_args()

    print(f"Loading model from {args.model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path)
    model.eval()

    template = load_chat_format(args.model_path)

    print("Tiny-LLM chat. Type 'exit' or 'quit' to stop.\n")
    while True:
        try:
            prompt = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if prompt.lower() in ("exit", "quit"):
            break
        if not prompt:
            continue

        reply = generate_text(prompt, model, tokenizer, template)
        print(f"Tiny-LLM: {reply}\n")


if __name__ == "__main__":
    main()
