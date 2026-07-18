"""
compare_performance.py
------------------------
Loads the result CSVs from zero-shot, few-shot, and fine-tuned runs and
prints/plots a side-by-side comparison of top-1 and top-3 accuracy.

Usage:
    python compare_performance.py
"""

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zero_shot_csv", type=str, default="./results/zero_shot_results.csv")
    parser.add_argument("--few_shot_csv", type=str, default="./results/few_shot_results.csv")
    parser.add_argument("--finetuned_csv", type=str, default="./results/finetuned_results.csv")
    parser.add_argument("--plot_path", type=str, default="./results/comparison.png")
    return parser.parse_args()


def load_accuracy(path, label):
    if not os.path.exists(path):
        print(f"  [skipped] {label}: no results file at {path} (run its script first)")
        return None
    df = pd.read_csv(path)
    top1 = df["top1_correct"].mean()
    top3 = df["top3_correct"].mean()
    return {"method": label, "top1_accuracy": top1, "top3_accuracy": top3}


def main():
    args = parse_args()

    print("Loading results...")
    results = []
    for path, label in [
        (args.zero_shot_csv, "Zero-shot"),
        (args.few_shot_csv, "Few-shot"),
        (args.finetuned_csv, "Fine-tuned"),
    ]:
        row = load_accuracy(path, label)
        if row:
            results.append(row)

    if not results:
        print("No results found. Run zero_shot_tagging.py / few_shot_tagging.py / finetune_classifier.py first.")
        return

    summary = pd.DataFrame(results)
    print("\nPerformance comparison:")
    print(summary.to_string(index=False, formatters={
        "top1_accuracy": "{:.2%}".format,
        "top3_accuracy": "{:.2%}".format,
    }))

    # Bar chart
    x = range(len(summary))
    width = 0.35
    plt.figure(figsize=(7, 5))
    plt.bar([i - width / 2 for i in x], summary["top1_accuracy"], width, label="Top-1 accuracy")
    plt.bar([i + width / 2 for i in x], summary["top3_accuracy"], width, label="Top-3 accuracy")
    plt.xticks(list(x), summary["method"])
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.title("Zero-shot vs Few-shot vs Fine-tuned")
    plt.legend()
    plt.tight_layout()

    os.makedirs(os.path.dirname(args.plot_path), exist_ok=True)
    plt.savefig(args.plot_path)
    print(f"\nSaved comparison chart to {args.plot_path}")


if __name__ == "__main__":
    main()
