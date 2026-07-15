"""
evaluate_model.py
------------------
Loads a fine-tuned model and reports accuracy, weighted F1, and a
per-class classification report on the AG News test split.

Usage:
    python evaluate_model.py --model_dir ./bert-agnews/final
"""

import argparse
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from sklearn.metrics import classification_report, confusion_matrix

LABEL_NAMES = ["World", "Sports", "Business", "Sci/Tech"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="./bert-agnews/final")
    parser.add_argument("--max_length", type=int, default=128)
    return parser.parse_args()


def main():
    args = parse_args()

    dataset = load_dataset("ag_news", split="test")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=args.max_length)

    tokenized = dataset.map(tokenize_function, batched=True)
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    trainer = Trainer(model=model, tokenizer=tokenizer)
    predictions = trainer.predict(tokenized)

    preds = np.argmax(predictions.predictions, axis=-1)
    labels = predictions.label_ids

    print("Classification report:")
    print(classification_report(labels, preds, target_names=LABEL_NAMES, digits=4))

    print("Confusion matrix:")
    print(confusion_matrix(labels, preds))


if __name__ == "__main__":
    main()
