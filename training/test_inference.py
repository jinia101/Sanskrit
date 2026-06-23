#!/usr/bin/env python3
"""Quick inference test for a fine-tuned LoRA adapter."""

from __future__ import annotations

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

INSTRUCTION = (
    "Given a Sanskrit verse line and a token, list all valid morphological analyses "
    "including lemma, part of speech, and grammatical tags."
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--adapter-dir", default="training/checkpoints/bhagavata-morph-lora")
    parser.add_argument("--verse-line", default="śrīśuka uvāca varīyān eṣa te praśnaḥ")
    parser.add_argument("--token", default="uvāca")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, args.adapter_dir)
    model.eval()

    messages = [
        {"role": "system", "content": INSTRUCTION},
        {
            "role": "user",
            "content": f"Verse line: {args.verse_line}\nToken: {args.token}",
        },
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
    print(tokenizer.decode(output[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
