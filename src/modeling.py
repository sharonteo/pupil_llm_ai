import pandas as pd
import numpy as np
from typing import Dict, Any

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve
)

from pathlib import Path

# Path to synthetic dataset
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "synthetic_pupillometry.csv"


# -----------------------------
# Feature configuration
# -----------------------------
FEATURES = [
    "age", "sex", "diagnosis", "npi",
    "true_pupil_size", "measured_pupil_size",
    "pupil_left", "pupil_right",
    "constriction_velocity", "dilation_velocity",
    "latency_ms"
]

TARGET = "gcs_severe"


# -----------------------------
# Data preparation
# -----------------------------
def _prepare_data(df: pd.DataFrame):
    df = df.copy()

    X = df[FEATURES]
    y = df[TARGET]

    # One-hot encode categorical variables
    X = pd.get_dummies(X, columns=["sex", "diagnosis"], drop_first=True)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale continuous features for LR
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler, X.columns


# -----------------------------
# Metrics computation
# -----------------------------
def _compute_metrics(y_true, y_prob, threshold=0.5) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "sensitivity": sensitivity,
        "specificity": specificity,
    }


# -----------------------------
# Model training
# -----------------------------
def train_models(df: pd.DataFrame) -> Dict[str, Any]:
    (
        X_train, X_test,
        X_train_scaled, X_test_scaled,
        y_train, y_test,
        scaler, feature_names
    ) = _prepare_data(df)

    models = {}

    # -------------------------
    # Logistic Regression
    # -------------------------
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train_scaled, y_train)
    y_prob_lr = lr.predict_proba(X_test_scaled)[:, 1]

    metrics_lr = _compute_metrics(y_test, y_prob_lr)
    fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)

    models["Logistic Regression"] = {
        "model": lr,
        "metrics": metrics_lr,
        "roc": (fpr_lr, tpr_lr),
    }

    # -------------------------
    # Random Forest
    # -------------------------
    rf = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42
    )
    rf.fit(X_train, y_train)
    y_prob_rf = rf.predict_proba(X_test)[:, 1]

    metrics_rf = _compute_metrics(y_test, y_prob_rf)
    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)

    models["Random Forest"] = {
        "model": rf,
        "metrics": metrics_rf,
        "roc": (fpr_rf, tpr_rf),
    }

    # -------------------------
    # XGBoost
    # -------------------------
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        eval_metric="logloss",
        use_label_encoder=False
    )
    xgb.fit(X_train, y_train)
    y_prob_xgb = xgb.predict_proba(X_test)[:, 1]

    metrics_xgb = _compute_metrics(y_test, y_prob_xgb)
    fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_prob_xgb)

    models["XGBoost"] = {
        "model": xgb,
        "metrics": metrics_xgb,
        "roc": (fpr_xgb, tpr_xgb),
    }

    return {
        "models": models,
        "y_test": y_test,
    }


# -----------------------------
# Convenience loader
# -----------------------------
def load_data_and_train():
    df = pd.read_csv(DATA_PATH)
    return df, train_models(df)