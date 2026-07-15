"""
train.py
--------
Fine-tunes bert-base-uncased on the AG News dataset for topic classification.

Labels (AG News):
    0 -> World
    1 -> Sports
    2 -> Business
    3 -> Sci/Tech

Usage:
    python train.py --epochs 3 --batch_size 16 --output_dir ./bert-agnews
"""

import argparse
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
import evaluate


LABEL_NAMES = ["World", "Sports", "Business", "Sci/Tech"]


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune BERT on AG News")
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--output_dir", type=str, default="./bert-agnews")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=128)
    # Use a subset for quick local testing; set to -1 to use the full dataset
    parser.add_argument("--train_subset", type=int, default=-1)
    parser.add_argument("--eval_subset", type=int, default=-1)
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Load dataset
    print("Loading AG News dataset...")
    dataset = load_dataset("ag_news")

    if args.train_subset > 0:
        dataset["train"] = dataset["train"].shuffle(seed=42).select(range(args.train_subset))
    if args.eval_subset > 0:
        dataset["test"] = dataset["test"].shuffle(seed=42).select(range(args.eval_subset))

    # 2. Tokenizer
    print(f"Loading tokenizer for {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=args.max_length,
        )

    print("Tokenizing dataset...")
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels"]
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 3. Model
    print(f"Loading model {args.model_name}...")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(LABEL_NAMES),
        id2label={i: name for i, name in enumerate(LABEL_NAMES)},
        label2id={name: i for i, name in enumerate(LABEL_NAMES)},
    )

    # 4. Metrics
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        acc = accuracy_metric.compute(predictions=predictions, references=labels)
        f1 = f1_metric.compute(predictions=predictions, references=labels, average="weighted")
        return {"accuracy": acc["accuracy"], "f1": f1["f1"]}

    # 5. Training arguments
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
        metric_for_best_model="f1",
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=50,
        report_to="none",
    )

    # 6. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # 7. Train
    print("Starting training...")
    trainer.train()

    # 8. Final evaluation
    print("Running final evaluation...")
    metrics = trainer.evaluate()
    print(f"Final metrics: {metrics}")

    # 9. Save model + tokenizer for deployment
    print(f"Saving model to {args.output_dir}/final...")
    trainer.save_model(f"{args.output_dir}/final")
    tokenizer.save_pretrained(f"{args.output_dir}/final")

    print("Done. Model ready for deployment.")


if __name__ == "__main__":
    main()
