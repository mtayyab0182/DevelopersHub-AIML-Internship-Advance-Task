"""
End-to-End ML Pipeline for Telco Customer Churn Prediction
=============================================================
Builds a reusable, production-ready scikit-learn Pipeline that:
  1. Preprocesses raw data (imputation, scaling, one-hot encoding)
  2. Trains/tunes Logistic Regression and Random Forest via GridSearchCV
  3. Selects the best model on held-out test data
  4. Exports the ENTIRE pipeline (preprocessing + model) as a single .joblib file

Usage:
    python churn_pipeline.py --data Telco-Customer-Churn.csv

The exported pipeline.joblib can be loaded later and called directly on
raw, unprocessed customer records — no manual preprocessing needed at
inference time.
"""

import argparse
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# 1. Data loading & light cleaning
# ---------------------------------------------------------------------------
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Known quirk in this dataset: TotalCharges is stored as a string with
    # some blank entries for customers with 0 tenure. Coerce to numeric.
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # customerID is a unique identifier -> no predictive value, drop it.
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    return df


# ---------------------------------------------------------------------------
# 2. Build the preprocessing + modeling pipeline
# ---------------------------------------------------------------------------
def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

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

    return preprocessor


def build_candidates(preprocessor: ColumnTransformer) -> dict:
    """
    Returns a dict of {name: (pipeline, param_grid)} — one full Pipeline
    (preprocessing + classifier) per model type, each with its own
    hyperparameter search space.
    """
    candidates = {}

    # --- Logistic Regression ---
    logreg_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
    ])
    logreg_grid = {
        "classifier__C": [0.01, 0.1, 1, 10],
        "classifier__penalty": ["l2"],
        "classifier__class_weight": [None, "balanced"],
    }
    candidates["LogisticRegression"] = (logreg_pipeline, logreg_grid)

    # --- Random Forest ---
    rf_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(random_state=RANDOM_STATE)),
    ])
    rf_grid = {
        "classifier__n_estimators": [200, 400],
        "classifier__max_depth": [None, 8, 16],
        "classifier__min_samples_leaf": [1, 2, 4],
        "classifier__class_weight": [None, "balanced"],
    }
    candidates["RandomForest"] = (rf_pipeline, rf_grid)

    return candidates


# ---------------------------------------------------------------------------
# 3. Train, tune, and select the best model
# ---------------------------------------------------------------------------
def run(data_path: str, output_path: str, cv: int = 5, n_jobs: int = -1):
    df = load_data(data_path)

    target_col = "Churn"
    X = df.drop(columns=[target_col])
    y = (df[target_col].astype(str).str.strip().str.lower() == "yes").astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    preprocessor = build_preprocessor(X_train)
    candidates = build_candidates(preprocessor)

    results = {}
    fitted_searches = {}

    for name, (pipeline, param_grid) in candidates.items():
        print(f"\n{'=' * 60}\nTuning {name}\n{'=' * 60}")
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=cv,
            scoring="f1",         # F1 is a sane default for churn (class imbalance)
            n_jobs=n_jobs,
            verbose=1,
        )
        search.fit(X_train, y_train)
        fitted_searches[name] = search

        y_pred = search.predict(X_test)
        y_proba = search.predict_proba(X_test)[:, 1]

        results[name] = {
            "best_params": search.best_params_,
            "cv_best_f1": search.best_score_,
            "test_accuracy": accuracy_score(y_test, y_pred),
            "test_f1": f1_score(y_test, y_pred),
            "test_roc_auc": roc_auc_score(y_test, y_proba),
        }

        print(f"Best params: {search.best_params_}")
        print(f"CV best F1:  {search.best_score_:.4f}")
        print(f"Test F1:     {results[name]['test_f1']:.4f}")
        print(f"Test AUC:    {results[name]['test_roc_auc']:.4f}")
        print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))
        print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

    # --- pick the best model by test F1 (swap for whatever metric matters most) ---
    best_name = max(results, key=lambda k: results[k]["test_f1"])
    best_pipeline = fitted_searches[best_name].best_estimator_

    print(f"\n{'=' * 60}\nSelected best model: {best_name}\n{'=' * 60}")
    for k, v in results[best_name].items():
        print(f"{k}: {v}")

    # --- export the FULL pipeline (preprocessing + trained model) ---
    joblib.dump(best_pipeline, output_path)
    print(f"\nSaved production-ready pipeline to: {output_path}")

    # also save a small results summary alongside it
    summary_path = str(Path(output_path).with_suffix("")) + "_results.csv"
    pd.DataFrame(results).T.to_csv(summary_path)
    print(f"Saved model comparison summary to: {summary_path}")

    return best_pipeline, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and export a churn prediction pipeline.")
    parser.add_argument("--data", type=str, default="Telco-Customer-Churn.csv",
                         help="Path to the Telco churn CSV file.")
    parser.add_argument("--output", type=str, default="churn_pipeline.joblib",
                         help="Path to save the exported pipeline.")
    parser.add_argument("--cv", type=int, default=5, help="Number of cross-validation folds.")
    args = parser.parse_args()

    run(args.data, args.output, cv=args.cv)
