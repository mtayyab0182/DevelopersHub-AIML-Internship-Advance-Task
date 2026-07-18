"""
zero_shot_tagging.py
---------------------
Tags each support ticket with its top-3 most likely categories using
zero-shot LLM prompting (no labeled examples given to the model).

Usage:
    python zero_shot_tagging.py --input_csv ./data/tickets.csv --llm_backend local --n 40
"""

import argparse
import pandas as pd
from tqdm import tqdm

from utils import CATEGORIES, build_zero_shot_prompt, parse_top3, build_llm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, default="./data/tickets.csv")
    parser.add_argument("--output_csv", type=str, default="./results/zero_shot_results.csv")
    parser.add_argument("--llm_backend", type=str, choices=["local", "openai"], default="local")
    parser.add_argument("--n", type=int, default=-1, help="Limit number of tickets (-1 = all)")
    return parser.parse_args()


def main():
    args = parse_args()

    df = pd.read_csv(args.input_csv)
    if args.n > 0:
        df = df.head(args.n)

    print(f"Loading {args.llm_backend} LLM backend...")
    generate = build_llm(llm_backend=args.llm_backend)

    predicted_tags = []
    print(f"Zero-shot tagging {len(df)} tickets...")
    for text in tqdm(df["text"]):
        prompt = build_zero_shot_prompt(text)
        response = generate(prompt)
        tags = parse_top3(response, CATEGORIES)
        predicted_tags.append(tags)

    df["predicted_tags"] = [", ".join(tags) for tags in predicted_tags]
    df["top1_correct"] = [
        (tags[0] == true) if tags else False
        for tags, true in zip(predicted_tags, df["true_category"])
    ]
    df["top3_correct"] = [
        true in tags for tags, true in zip(predicted_tags, df["true_category"])
    ]

    top1_acc = df["top1_correct"].mean()
    top3_acc = df["top3_correct"].mean()
    print(f"\nZero-shot Top-1 accuracy: {top1_acc:.2%}")
    print(f"Zero-shot Top-3 accuracy: {top3_acc:.2%}")

    import os
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(f"Saved results to {args.output_csv}")


if __name__ == "__main__":
    main()
