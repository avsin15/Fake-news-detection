"""
model_retrain.py
----------------
Retrains Fake News / Factuality detection models using the unified dataset.

Steps:
1. Load factual_dataset_clean.csv
2. Encode text using SentenceTransformer
3. Train Logistic Regression and XGBoost models
4. Evaluate and compare
5. Save best-performing model + metrics

Output:
- models/logistic_regression.pkl
- models/xgboost_model.pkl
- models/training_metrics.json
- models/plots/*.png
"""

import os
import json
import joblib
import logging
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer

# ============================================================
# CONFIGURATION
# ============================================================
DATA_PATH = "data/processed/factual_dataset_clean.csv"
MODEL_DIR = "models"
PLOTS_DIR = os.path.join(MODEL_DIR, "plots")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

MODEL_NAME = "all-MiniLM-L6-v2"
TEST_SIZE = 0.2
VAL_SIZE = 0.1
RANDOM_STATE = 42
BATCH_SIZE = 64

# ============================================================
# LOGGING
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ============================================================
# DATA LOADING & PREPROCESSING
# ============================================================

def load_data():
    """Load processed factual dataset."""
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"❌ Processed dataset not found at {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["text", "label"])
    logger.info(f"✅ Loaded dataset: {len(df)} samples")
    return df


def encode_labels(df):
    """Encode text labels into binary."""
    label_map = {"real": 0, "fake": 1, "partially_true": 2}
    df["label_encoded"] = df["label"].map(label_map)
    df = df.dropna(subset=["label_encoded"])
    return df, label_map


def get_embeddings(texts, model_name=MODEL_NAME, batch_size=BATCH_SIZE):
    """Generate dense embeddings using SentenceTransformer."""
    model = SentenceTransformer(model_name)
    logger.info(f"⚙️ Generating embeddings with {model_name}...")
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)
    logger.info(f"✅ Embeddings generated with shape {embeddings.shape}")
    return embeddings


def split_data(X, y):
    """Train/Val/Test split."""
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(TEST_SIZE + VAL_SIZE), random_state=RANDOM_STATE, stratify=y
    )
    val_ratio = VAL_SIZE / (TEST_SIZE + VAL_SIZE)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1 - val_ratio), random_state=RANDOM_STATE, stratify=y_temp
    )
    logger.info(f"Split sizes: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test

# ============================================================
# MODEL TRAINING FUNCTIONS
# ============================================================

def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs",
        multi_class="auto",
        n_jobs=-1
    )
    logger.info("⚙️ Training Logistic Regression...")
    model.fit(X_train, y_train)
    logger.info("✅ Logistic Regression training complete.")
    return model


def train_xgboost(X_train, y_train):
    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        use_label_encoder=False
    )
    logger.info("⚙️ Training XGBoost model...")
    model.fit(X_train, y_train)
    logger.info("✅ XGBoost training complete.")
    return model
"""from sklearn.calibration import CalibratedClassifierCV

# for Logistic Regression
base_model = LogisticRegression(max_iter=1000, class_weight="balanced")
calibrated_model = CalibratedClassifierCV(base_model, cv=3, method="isotonic")
calibrated_model.fit(X_train, y_train)
joblib.dump(calibrated_model, LOGISTIC_MODEL_PATH)"""



# ============================================================
# EVALUATION & PLOTTING
# ============================================================

def evaluate_model(model, X_test, y_test, label_map, model_name):
    y_pred = model.predict(X_test)
    y_proba = (
        model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
    )

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "f1": f1_score(y_test, y_pred, average="macro", zero_division=0)
    }

    if y_proba is not None and y_proba.shape[1] == 2:
        metrics["auc_roc"] = roc_auc_score(y_test, y_proba[:, 1])
    else:
        metrics["auc_roc"] = None

    logger.info(f"✅ {model_name} Performance:")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v:.4f}" if v is not None else f"  {k}: N/A")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    labels = list(label_map.keys())

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels
    )
    plt.title(f"{model_name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{model_name}_confusion_matrix.png"), dpi=300)
    plt.close()

    return metrics


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    logger.info("🚀 Starting Model Retraining Pipeline...")

    df = load_data()
    df, label_map = encode_labels(df)

    X = get_embeddings(df["text"].tolist())
    y = df["label_encoded"].values

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # Train models
    log_model = train_logistic_regression(X_train, y_train)
    xgb_model = train_xgboost(X_train, y_train)

    # Evaluate
    metrics_log = evaluate_model(log_model, X_test, y_test, label_map, "LogisticRegression")
    metrics_xgb = evaluate_model(xgb_model, X_test, y_test, label_map, "XGBoost")

    # Save models
    joblib.dump(log_model, os.path.join(MODEL_DIR, "logistic_regression.pkl"))
    joblib.dump(xgb_model, os.path.join(MODEL_DIR, "xgboost_model.pkl"))

    # Save metrics
    results = {
        "timestamp": datetime.now().isoformat(),
        "model_used": MODEL_NAME,
        "metrics": {
            "LogisticRegression": metrics_log,
            "XGBoost": metrics_xgb
        }
    }
    with open(os.path.join(MODEL_DIR, "training_metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    logger.info("💾 Models and metrics saved successfully.")
    logger.info(f"📊 Best model: {'XGBoost' if metrics_xgb['f1'] > metrics_log['f1'] else 'Logistic Regression'}")
    logger.info("✅ Model retraining complete!")


if __name__ == "__main__":
    main()
