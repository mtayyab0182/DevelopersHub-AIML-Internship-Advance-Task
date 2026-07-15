<<<<<<< HEAD
# 📰 News Topic Classifier Using BERT

Fine-tunes `bert-base-uncased` on the [AG News dataset](https://huggingface.co/datasets/ag_news) to classify news headlines into four topics: **World, Sports, Business, Sci/Tech**. Includes evaluation (accuracy + F1) and a live demo app built with Streamlit or Gradio.

## Demo

![demo-placeholder](https://via.placeholder.com/800x400?text=App+Screenshot+Goes+Here)

## Project structure

```
news-topic-classifier-bert/
├── train.py             # Fine-tuning script (Hugging Face Trainer)
├── evaluate_model.py     # Standalone evaluation + classification report
├── app.py                # Streamlit deployment app
├── app_gradio.py          # Gradio deployment app (alternative)
├── requirements.txt
├── .gitignore
└── README.md
```

## Skills demonstrated

- NLP with Hugging Face Transformers
- Transfer learning / fine-tuning a pretrained BERT model
- Evaluation metrics for text classification (accuracy, weighted F1)
- Lightweight model deployment (Streamlit / Gradio)

## 1. Setup

```bash
git clone https://github.com/<your-username>/news-topic-classifier-bert.git
cd news-topic-classifier-bert

python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

A GPU is strongly recommended for training (Colab's free T4 works fine). CPU-only training will run but is slow.

## 2. Dataset

The AG News dataset loads automatically from Hugging Face Datasets the first time you run `train.py` — no manual download needed.

| Split | Examples | Classes |
|---|---|---|
| train | 120,000 | 4 |
| test  | 7,600   | 4 |

Labels: `0=World, 1=Sports, 2=Business, 3=Sci/Tech`

## 3. Fine-tune the model

```bash
python train.py --epochs 3 --batch_size 16 --output_dir ./bert-agnews
```

Useful flags:

| Flag | Default | Purpose |
|---|---|---|
| `--epochs` | 3 | Training epochs |
| `--batch_size` | 16 | Per-device batch size |
| `--lr` | 2e-5 | Learning rate |
| `--train_subset` | -1 (full) | Use a small subset for a quick smoke test, e.g. `2000` |
| `--eval_subset` | -1 (full) | Same, for the eval split |

The best checkpoint (by weighted F1) is saved to `./bert-agnews/final`.

For a fast sanity check before a full run:

```bash
python train.py --train_subset 2000 --eval_subset 500 --epochs 1
```

## 4. Evaluate

```bash
python evaluate_model.py --model_dir ./bert-agnews/final
```

Prints accuracy, weighted F1, a per-class precision/recall/F1 report, and a confusion matrix. Typical results for `bert-base-uncased` fine-tuned for 3 epochs on full AG News are ~94–95% accuracy.

## 5. Deploy

**Streamlit:**

```bash
streamlit run app.py
```

**Gradio** (alternative):

```bash
python app_gradio.py
```

Both apps load the model from `./bert-agnews/final` — update `MODEL_DIR` in `app.py` / `app_gradio.py` if you saved it elsewhere.

## Results

| Metric | Score |
|---|---|
| Accuracy | *fill in after training* |
| Weighted F1 | *fill in after training* |

## License

MIT
=======
# DevelopersHub-AI-ML-Advanced-Internship-Tasks
Build hands-on ML and AI skills through projects on transformer models, ML pipelines, multimodal learning, conversational AI, and LLM applications. Gain practical, job-ready experience using Hugging Face, scikit-learn, LangChain, Streamlit, Gradio, CNNs, and joblib to create a strong technical portfolio for industry roles.
>>>>>>> d41b5b70f7e82205d65b20afb9e8d7bc3597f465
