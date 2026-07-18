"""
predict_ticket.py
--------------------
Tags a single new support ticket with its top-3 categories, using either
the fine-tuned classifier (recommended, most accurate) or zero-shot/few-shot
prompting.

Usage (fine-tuned, after running finetune_classifier.py):
    python predict_ticket.py --text "My payment failed twice this week" --method finetuned

Usage (zero-shot, no training needed):
    python predict_ticket.py --text "My payment failed twice this week" --method zero_shot
"""

import argparse
import numpy as np
import torch

from utils import CATEGORIES, build_zero_shot_prompt, build_few_shot_prompt, parse_top3, build_llm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument(
        "--method", type=str, choices=["zero_shot", "few_shot", "finetuned"], default="finetuned"
    )
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/finetuned/final")
    parser.add_argument("--llm_backend", type=str, choices=["local", "openai"], default="local")
    return parser.parse_args()


def predict_finetuned(text, checkpoint):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
    model.eval()

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1).squeeze().numpy()

    top3_idx = np.argsort(-probs)[:3]
    id2label = model.config.id2label
    return [(id2label[int(i)], float(probs[i])) for i in top3_idx]


def predict_prompted(text, method, llm_backend):
    generate = build_llm(llm_backend=llm_backend)
    prompt = build_zero_shot_prompt(text) if method == "zero_shot" else build_few_shot_prompt(text)
    response = generate(prompt)
    tags = parse_top3(response, CATEGORIES)
    return [(tag, None) for tag in tags]


def main():
    args = parse_args()

    if args.method == "finetuned":
        results = predict_finetuned(args.text, args.checkpoint)
    else:
        results = predict_prompted(args.text, args.method, args.llm_backend)

    print(f"\nTicket: {args.text}")
    print(f"Method: {args.method}\n")
    print("Top 3 predicted tags:")
    for rank, (tag, score) in enumerate(results, start=1):
        if score is not None:
            print(f"  {rank}. {tag} ({score:.1%})")
        else:
            print(f"  {rank}. {tag}")


if __name__ == "__main__":
    main()
