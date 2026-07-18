"""
finetune_classifier.py
------------------------
Fine-tunes a small transformer (distilbert-base-uncased) directly on the
labeled ticket dataset for comparison against zero-shot/few-shot prompting.
Reports top-1 and top-3 accuracy on a held-out test split.

Usage:
    python finetune_classifier.py --input_csv ./data/tickets.csv --epochs 4
"""

import argparse
import os
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)

from utils import CATEGORIES


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, default="./data/tickets.csv")
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/finetuned")
    parser.add_argument("--results_csv", type=str, default="./results/finetuned_results.csv")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()

    label2id = {cat: i for i, cat in enumerate(CATEGORIES)}
    id2label = {i: cat for cat, i in label2id.items()}

    df = pd.read_csv(args.input_csv)
    df["label"] = df["true_category"].map(label2id)

    train_df, test_df = train_test_split(
        df, test_size=args.test_size, stratify=df["label"], random_state=args.seed
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=128)

    train_ds = Dataset.from_pandas(train_df[["text", "label"]].reset_index(drop=True)).map(
        tokenize, batched=True
    )
    test_ds = Dataset.from_pandas(test_df[["text", "label"]].reset_index(drop=True)).map(
        tokenize, batched=True
    )

    train_ds = train_ds.rename_column("label", "labels")
    test_ds = test_ds.rename_column("label", "labels")
    train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    test_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(CATEGORIES),
        id2label=id2label,
        label2id=label2id,
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        logging_steps=20,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    print("Fine-tuning classifier on labeled tickets...")
    trainer.train()

    print("Saving fine-tuned model...")
    trainer.save_model(f"{args.output_dir}/final")
    tokenizer.save_pretrained(f"{args.output_dir}/final")

    # Evaluate with top-1 / top-3 accuracy using predicted probabilities
    print("Evaluating on held-out test set...")
    predictions = trainer.predict(test_ds)
    logits = predictions.predictions
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()

    top1_preds = np.argmax(probs, axis=-1)
    top3_preds = np.argsort(-probs, axis=-1)[:, :3]

    true_labels = test_df["label"].values
    top1_correct = top1_preds == true_labels
    top3_correct = np.array(
        [true_labels[i] in top3_preds[i] for i in range(len(true_labels))]
    )

    top1_acc = top1_correct.mean()
    top3_acc = top3_correct.mean()
    print(f"\nFine-tuned Top-1 accuracy: {top1_acc:.2%}")
    print(f"Fine-tuned Top-3 accuracy: {top3_acc:.2%}")

    results_df = test_df.copy().reset_index(drop=True)
    results_df["predicted_tags"] = [
        ", ".join(id2label[idx] for idx in row) for row in top3_preds
    ]
    results_df["top1_correct"] = top1_correct
    results_df["top3_correct"] = top3_correct

    os.makedirs(os.path.dirname(args.results_csv), exist_ok=True)
    results_df.to_csv(args.results_csv, index=False)
    print(f"Saved results to {args.results_csv}")


if __name__ == "__main__":
    main()
