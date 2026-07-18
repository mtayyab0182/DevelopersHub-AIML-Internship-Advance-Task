# DevelopersHub AI/ML Advanced Internship — Tasks

Five hands-on projects covering the AI/ML lifecycle end to end: data
preprocessing, model development (transformers, CNNs, classical ML,
LLMs/RAG), evaluation, and deployment. Each task lives in its own folder
with a runnable pipeline, its own README (objective, methodology, results),
and a Streamlit/CLI demo where applicable.

## Tasks at a glance

| # | Task | Core techniques | Key metrics |
|---|---|---|---|
| 1 | [News Topic Classifier Using BERT](<./Task%201%20News%20Topic%20Classifier%20Using%20BERT>) | Transfer learning, transformer fine-tuning | Accuracy, weighted F1 |
| 2 | [End-to-End ML Pipeline with Scikit-learn](<./Task%202%20End%20to%20End%20ML%20Pipeline%20with%20Scikit%20learn%20Pipeline%20API>) | `Pipeline`/`ColumnTransformer`, hyperparameter tuning | Accuracy, precision, recall, F1, ROC-AUC |
| 3 | [Multimodal Housing Price Prediction](<./Task%203%20Multimodal%20ML%20Housing%20Price%20Prediction%20Using%20Images%20Tabular%20Data>) | CNN + tabular feature fusion | MAE, RMSE |
| 4 | [Context-Aware Chatbot Using LangChain / RAG](<./Task%204%20Context%20Aware%20Chatbot%20Using%20LangChain%20or%20RAG>) | Vector search (FAISS), conversational memory | Retrieval quality, source grounding |
| 5 | [Auto-Tagging Support Tickets Using LLM](<./Task%205%20Auto%20Tagging%20Support%20Tickets%20Using%20LLM>) | Zero-shot / few-shot prompting, fine-tuning | Top-1 / top-3 accuracy |

## Task summaries

### 1. News Topic Classifier Using BERT
Fine-tunes `bert-base-uncased` on the AG News dataset to classify headlines
into **World, Sports, Business, Sci/Tech**. Tokenizes and preprocesses with
Hugging Face `datasets`/`transformers`, trains with the `Trainer` API,
evaluates with accuracy and weighted F1, and deploys as a live Streamlit or
Gradio demo.
**Stack**: BERT, Hugging Face Transformers, Streamlit/Gradio

### 2. End-to-End ML Pipeline Using Scikit-learn
Builds a single, reusable Scikit-learn `Pipeline` (preprocessing +
`RandomForestClassifier`) for customer churn prediction, so preprocessing
and modeling travel together as one object. Tunes hyperparameters with
`GridSearchCV`, evaluates with accuracy/precision/recall/F1/ROC-AUC, and
exports the fitted pipeline with `joblib` for direct inference on raw
records.
**Stack**: Scikit-learn, `ColumnTransformer`, `GridSearchCV`, joblib

### 3. Multimodal ML — Housing Price Prediction
Predicts housing prices by fusing a CNN image encoder (pretrained ResNet18)
with a tabular MLP encoder (sqft, bedrooms, bathrooms, lot size, year
built), concatenating both embeddings before a regression head. Evaluated
with MAE and RMSE, plus a predicted-vs-actual scatter plot.
**Stack**: PyTorch, torchvision (ResNet18), scikit-learn

### 4. Context-Aware Chatbot Using LangChain / RAG
A conversational chatbot that retrieves answers from a vectorized document
store (FAISS) and remembers conversation history via
`ConversationBufferMemory`, combined through a
`ConversationalRetrievalChain`. Deployed with Streamlit, showing which
source documents backed each answer. Supports a fully local LLM backend
(`flan-t5-base`) or OpenAI.
**Stack**: LangChain, FAISS, sentence-transformers, Streamlit

### 5. Auto-Tagging Support Tickets Using LLM
Tags free-text support tickets with their top-3 most likely categories,
comparing **zero-shot prompting**, **few-shot prompting**, and a
**fine-tuned DistilBERT classifier** head to head on the same dataset, with
a side-by-side accuracy comparison chart.
**Stack**: Transformers (DistilBERT), prompt engineering, scikit-learn

## Repository structure

```
DevelopersHub-AIML-Internship-Advance-Task/
├── Task 1 News Topic Classifier Using BERT/
├── Task 2 End to End ML Pipeline with Scikit learn Pipeline API/
├── Task 3 Multimodal ML Housing Price Prediction Using Images Tabular Data/
├── Task 4 Context Aware Chatbot Using LangChain or RAG/
├── Task 5 Auto Tagging Support Tickets Using LLM/
└── README.md   <- you are here
```

Each task folder is self-contained: its own `requirements.txt`, `.gitignore`,
and `README.md` with Objective / Methodology / Key Results sections, so any
task can be set up and run independently of the others.

## Running any task

Every task follows the same pattern:

```bash
cd "Task N <name>"
python -m venv .venv
.venv\Scripts\activate        # Windows; use `source .venv/bin/activate` on Mac/Linux
pip install -r requirements.txt
```

Then follow that task's own README for its specific data-generation,
training, evaluation, and deployment commands.

## Skills covered across all 5 tasks

- NLP & transfer learning (BERT, DistilBERT fine-tuning)
- Classical ML pipelines & hyperparameter tuning (Scikit-learn)
- Computer vision & multimodal fusion (CNNs + tabular data)
- Retrieval-Augmented Generation & conversational memory (LangChain, FAISS)
- Prompt engineering (zero-shot / few-shot) vs supervised fine-tuning
- Model evaluation across task types (classification, regression, ranking)
- Lightweight deployment (Streamlit, Gradio)

## Author

Muhammad Tayyab — [GitHub](https://github.com/mtayyab0182)
