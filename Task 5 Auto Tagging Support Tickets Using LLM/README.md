# 🏷️ Task 5: Auto Tagging Support Tickets Using LLM

Automatically tags free-text support tickets with their top-3 most likely
categories, comparing three approaches: **zero-shot prompting**, **few-shot
prompting**, and a **fine-tuned classifier**.

## Categories

```
Billing · Technical Issue · Account Access · Feature Request
Bug Report · General Inquiry · Refund Request · Shipping and Delivery
```

## Project structure

```
Task-5-Auto-Tagging-Support-Tickets-Using-LLM/
├── generate_synthetic_tickets.py   # Builds a labeled test dataset — no download needed
├── utils.py                         # Categories, prompt templates, response parsing, LLM backend
├── zero_shot_tagging.py             # Tags tickets with no labeled examples in the prompt
├── few_shot_tagging.py              # Tags tickets with a few labeled examples in the prompt
├── finetune_classifier.py           # Fine-tunes DistilBERT directly on labeled tickets
├── compare_performance.py           # Compares top-1 / top-3 accuracy across all 3 methods
├── predict_ticket.py                # Tag a single new ticket
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## Skills demonstrated

- Prompt engineering (zero-shot and few-shot)
- LLM-based text classification
- Zero-shot vs few-shot vs fine-tuned comparison
- Multi-class prediction and ranking (top-3 tags per ticket)

## 1. Setup

```bash
cd Task-5-Auto-Tagging-Support-Tickets-Using-LLM
python -m venv .venv
.venv\Scripts\activate        # Windows; use `source .venv/bin/activate` on Mac/Linux
pip install -r requirements.txt
```

### (Optional) OpenAI backend
By default, zero-shot/few-shot tagging use a **local** model (`google/flan-t5-base`)
— no API key needed. For higher-quality prompting results:
```bash
copy .env.example .env      # Windows; `cp .env.example .env` on Mac/Linux
```
Add your key to `.env`:
```
OPENAI_API_KEY=sk-...
```

## 2. Generate the dataset

```bash
python generate_synthetic_tickets.py --n_per_category 40 --output_csv ./data/tickets.csv
```
Creates 320 labeled synthetic tickets (40 per category) at `./data/tickets.csv`
with columns `id, text, true_category`. Swap in a real free-text ticket dataset
by matching this same schema.

## 3. Run zero-shot tagging

```bash
python zero_shot_tagging.py --input_csv ./data/tickets.csv --llm_backend local --n 40
```
`--n 40` limits it to 40 tickets for a fast first test — drop the flag to run
the full set. Prints top-1/top-3 accuracy and saves per-ticket predictions to
`./results/zero_shot_results.csv`.

## 4. Run few-shot tagging

```bash
python few_shot_tagging.py --input_csv ./data/tickets.csv --llm_backend local --n 40
```
Same as above, but the prompt includes 8 labeled examples (one per category)
before asking about the new ticket. Saves to `./results/few_shot_results.csv`.

## 5. Fine-tune a classifier for comparison

```bash
python finetune_classifier.py --input_csv ./data/tickets.csv --epochs 4
```
Fine-tunes `distilbert-base-uncased` on 80% of the tickets, evaluates top-1/top-3
accuracy on the held-out 20%, and saves the model to `./checkpoints/finetuned/final`.
Results save to `./results/finetuned_results.csv`.

## 6. Compare all three methods

```bash
python compare_performance.py
```
Prints a side-by-side accuracy table and saves a bar chart to
`./results/comparison.png`.

Typical pattern you should expect to see: fine-tuning outperforms both prompting
methods once there's enough labeled data, few-shot outperforms zero-shot, and the
gap narrows as the local prompting model is swapped for a stronger one (e.g. OpenAI).

## 7. Tag a single new ticket

```bash
python predict_ticket.py --text "My payment failed twice this week" --method finetuned
```
```
python predict_ticket.py --text "My payment failed twice this week" --method zero_shot
```

## Results

| Method | Top-1 Accuracy | Top-3 Accuracy |
|---|---|---|
| Zero-shot | *fill in after running* | *fill in after running* |
| Few-shot | *fill in after running* | *fill in after running* |
| Fine-tuned | *fill in after running* | *fill in after running* |

## Notes

- `google/flan-t5-base` is a small, general-purpose model — it's a reasonable
  free stand-in for prompting experiments, but a stronger model (via
  `--llm_backend openai`) will show a bigger zero-shot/few-shot gap and higher
  overall accuracy.
- `parse_top3()` in `utils.py` is intentionally forgiving about formatting
  since generative models don't always follow output instructions exactly —
  it fuzzy-matches responses against the known category list.
- For a real dataset, keep the same CSV schema (`text`, `true_category`) so
  every script here works unmodified.

## License

MIT
