import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "../models/Tiny-LLM"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)


def generate_text(prompt, model, tokenizer, max_length=512, temperature=1.0, top_k=50, top_p=0.95):
    inputs = tokenizer.encode(prompt, return_tensors="pt")

    outputs = model.generate(
        inputs,
        max_length=max_length,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        do_sample=True,
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def main():
    prompt = "According to all known laws of aviation, there is no way a bee should be able to fly."
    print(generate_text(prompt, model, tokenizer))


if __name__ == "__main__":
    main()
