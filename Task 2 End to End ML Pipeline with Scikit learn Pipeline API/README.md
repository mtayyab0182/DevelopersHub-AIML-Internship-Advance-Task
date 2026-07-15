
# 📉 Customer Churn Prediction — End-to-End ML Pipeline

A reusable, production-ready machine learning pipeline built with **scikit-learn's `Pipeline` API** to predict customer churn on the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn). Includes hyperparameter tuning with `GridSearchCV`, model export via `joblib`, and a live **Streamlit** demo app.


## 🎯 Objective

Build a churn-prediction pipeline that is:
- **Reusable** — one object handles preprocessing *and* prediction, so it can be dropped into any script or app without re-writing feature engineering.
- **Production-ready** — trained, tuned, evaluated, and exported as a single deployable artifact.


## 🗂️ Project Structure

churn-app/
├── app.py                       # Streamlit demo app
├── churn_pipeline.py            # Training script (build, tune, evaluate, export)
├── predict_example.py           # Example: load the exported pipeline for inference
├── churn_pipeline.joblib        # Exported, trained pipeline (preprocessing + model)
├── churn_pipeline_results.csv   # Model comparison metrics (LogReg vs RandomForest)
├── Telco-Customer-Churn.csv     # Dataset
├── requirements.txt             # Python dependencies
└── README.md




## ✅ How Each Objective Was Implemented

### 1. Data preprocessing (scaling + encoding) using `Pipeline`

Numeric and categorical columns need different treatment, so a `ColumnTransformer` wraps two small pipelines — one per feature type — and that whole transformer becomes **step one** of the final model pipeline:

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features),
])
```

Why it matters: `handle_unknown="ignore"` means the pipeline won't crash in production if it sees a category it never saw during training. The imputers mean missing values (e.g. `TotalCharges` blanks for brand-new customers) are handled automatically, not manually.

### 2. Train Logistic Regression and Random Forest

Each model gets its own **full** pipeline (preprocessing + classifier bundled together), so preprocessing is always applied identically at train and inference time:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

logreg_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=2000, random_state=42)),
])

rf_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(random_state=42)),
])
```

### 3. Hyperparameter tuning with `GridSearchCV`

Each pipeline is wrapped in a grid search over its own hyperparameter space, scored on F1 (churn is an imbalanced class — accuracy alone would be misleading):

```python
from sklearn.model_selection import GridSearchCV

logreg_grid = {
    "classifier__C": [0.01, 0.1, 1, 10],
    "classifier__penalty": ["l2"],
    "classifier__class_weight": [None, "balanced"],
}

search = GridSearchCV(
    estimator=logreg_pipeline,
    param_grid=logreg_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1,
)
search.fit(X_train, y_train)
```

The same pattern runs for Random Forest with its own grid (`n_estimators`, `max_depth`, `min_samples_leaf`, `class_weight`). After both grids finish, the script compares test-set F1 across model types and keeps the winner.

### 4. Export the complete pipeline with `joblib`

Because preprocessing lives *inside* the pipeline object, exporting it exports everything — no separate scaler/encoder files to keep track of:

```python
import joblib
joblib.dump(best_pipeline, "churn_pipeline.joblib")
```

Loading it later for inference needs zero preprocessing code:

```python
pipeline = joblib.load("churn_pipeline.joblib")
prediction = pipeline.predict(new_raw_customer_dataframe)
```

---

## 📊 Results

| Model | CV Best F1 | Test F1 | Test ROC-AUC |
|---|---|---|---|
| Logistic Regression | 0.633 | 0.618 | 0.841 |
| Random Forest | 0.636 | 0.625 | 0.842 |

Random Forest was selected as the production model. Full metrics (precision/recall per class, best hyperparameters) are printed by `churn_pipeline.py` and saved to `churn_pipeline_results.csv`.

---

## 🚀 Setup & Usage (Local)

### 1. Clone the repo
```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```

### 2. Create a virtual environment and install dependencies
```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Train the pipeline (optional — a trained pipeline is already included)
```bash
python churn_pipeline.py --data Telco-Customer-Churn.csv --output churn_pipeline.joblib --cv 5
```

### 4. Run inference on new data
```bash
python predict_example.py
```

### 5. Launch the Streamlit demo app
```bash
streamlit run app.py
```
Then open the local URL it prints (usually `http://localhost:8501`).

---

## 🌐 Deploying the App (Streamlit Community Cloud)

1. Push this repo to GitHub (see below).
2. Go to **[share.streamlit.io](https://share.streamlit.io)** and sign in with GitHub.
3. Click **New app** → select your repo, branch `main`, main file `app.py`.
4. Click **Deploy**. Streamlit Cloud installs everything from `requirements.txt` automatically.
5. You'll get a public link like `https://<your-app-name>.streamlit.app`.

---

## 📤 Setting This Up on GitHub — Step by Step

### Step 1: Create a new repository on GitHub
1. Go to [github.com/new](https://github.com/new).
2. Name it (e.g. `churn-prediction-pipeline`).
3. Leave it **public** (required for the free tier of Streamlit Cloud to access it).
4. Don't initialize with a README (you already have one) — click **Create repository**.

### Step 2: Push your local files
From the folder containing all the project files:
```bash
git init
git add .
git commit -m "Initial commit: churn prediction pipeline + Streamlit app"
git branch -M main
git remote add origin https://github.com/<your-username>/<repo-name>.git
git push -u origin main
```

### Step 3: Verify
Refresh your GitHub repo page — you should see `app.py`, `churn_pipeline.py`, `churn_pipeline.joblib`, `requirements.txt`, and this `README.md` all listed.

### Step 4 (optional): Add a `.gitignore`
If you regenerate a virtual environment locally, keep it out of the repo:
```
venv/
__pycache__/
*.pyc
```

---

## 🧠 Skills Demonstrated

- ML pipeline construction with `sklearn.pipeline.Pipeline` and `ColumnTransformer`
- Hyperparameter tuning with `GridSearchCV`
- Model comparison and selection on held-out test data
- Model export and reuse with `joblib`
- Production-readiness: single-artifact deployment, `handle_unknown="ignore"` for robustness, a working demo app

# DevelopersHub-AIML-Internship-Advance-Task

