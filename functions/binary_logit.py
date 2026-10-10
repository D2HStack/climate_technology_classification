"""binary_logit.py

Logistic Regression hyperparameter tuning, model training, and evaluation
aligned with logistic_regression.py.
"""

from typing import Any, Dict, Optional, Union
import ast
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedShuffleSplit


# ==============================================================================
# 1. Feature Extraction & Metric Helpers
# ==============================================================================


def _extract_x_y(
    df: pd.DataFrame, y_label: str, x_label: str
) -> tuple[np.ndarray, np.ndarray]:
    """Extracts raw stacked embeddings X and binary label array y."""
    clean_df = df.dropna(subset=[x_label]).copy()

    raw_embeddings = clean_df[x_label].values
    if isinstance(raw_embeddings[0], str):
        embeddings_list = [ast.literal_eval(vec) for vec in raw_embeddings]
        X = np.vstack(embeddings_list).astype(np.float64)
    else:
        X = np.vstack(raw_embeddings).astype(np.float64)

    y = clean_df[y_label].astype(int).to_numpy()
    return X, y


def _compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray
) -> Dict[str, float]:
    """Computes exact metrics identical to logistic_regression.py."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    try:
        roc_val = float(roc_auc_score(y_true, y_proba))
    except (ValueError, IndexError):
        roc_val = np.nan

    try:
        prec_arr, rec_arr, _ = precision_recall_curve(y_true, y_proba)
        pr_val = float(auc(rec_arr, prec_arr))
    except (ValueError, IndexError):
        pr_val = np.nan

    return {
        "roc_auc": round(roc_val, 4) if pd.notna(roc_val) else np.nan,
        "pr_auc": round(pr_val, 4) if pd.notna(pr_val) else np.nan,
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "balanced_accuracy": round(
            float(balanced_accuracy_score(y_true, y_pred)), 4
        ),
        "sensitivity": round(sens, 4),
        "specificity": round(spec, 4),
        "precision": round(
            float(precision_score(y_true, y_pred, zero_division=0)), 4
        ),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
    }


# ==============================================================================
# 2. Grid Search over C and Tau (Repeated Stratified Splits)
# ==============================================================================


def evaluate_binary_logit_grid(
    df: pd.DataFrame,
    y_label: str,
    x_label: str,
    c_min: float,
    c_max: float,
    c_increment: float,
    tau_min: float,
    tau_max: float,
    tau_increment: float,
    split: float = 0.20,
    n_rep: int = 10,
    class_weight: Optional[Union[str, dict]] = "balanced",
    random_state: int = 42,
    max_iter: int = 1000,
) -> pd.DataFrame:
    """Evaluates Logistic Regression over a grid of C and tau matching

    logistic_regression.py solver and metric configurations.
    """
    if c_min <= 0:
        raise ValueError("c_min must be strictly greater than 0.")

    X, y = _extract_x_y(df, y_label, x_label)

    eps = 1e-9
    c_values = np.arange(c_min, c_max + eps, c_increment)
    tau_values = np.arange(tau_min, tau_max + eps, tau_increment)

    sss = StratifiedShuffleSplit(
        n_splits=n_rep, test_size=split, random_state=random_state
    )
    splits = list(sss.split(X, y))

    results = []

    for c_val in c_values:
        rep_data = []

        for train_idx, test_idx in splits:
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Fit unscaled lbfgs logistic regression (identical to logistic_regression.py)
            clf = LogisticRegression(
                C=float(c_val),
                class_weight=class_weight,
                solver="lbfgs",
                max_iter=max_iter,
                random_state=random_state,
            )
            clf.fit(X_train, y_train)

            y_probs = clf.predict_proba(X_test)[:, 1]
            rep_data.append((y_test, y_probs))

        for tau in tau_values:
            acc_list, spec_list, prec_list, rec_list, f1_list = [], [], [], [], []
            roc_auc_list, pr_auc_list = [], []

            for y_test, y_probs in rep_data:
                y_pred = (y_probs >= tau).astype(int)
                m = _compute_metrics(y_test, y_pred, y_probs)

                acc_list.append(m["accuracy"])
                spec_list.append(m["specificity"])
                prec_list.append(m["precision"])
                rec_list.append(m["recall"])
                f1_list.append(m["f1"])
                roc_auc_list.append(m["roc_auc"])
                pr_auc_list.append(m["pr_auc"])

            results.append(
                {
                    "C": round(float(c_val), 5),
                    "tau": round(float(tau), 4),
                    "Accuracy": np.nanmean(acc_list),
                    "Specificity": np.nanmean(spec_list),
                    "Precision": np.nanmean(prec_list),
                    "Recall": np.nanmean(rec_list),
                    "F1-Score": np.nanmean(f1_list),
                    "ROC-AUC": np.nanmean(roc_auc_list),
                    "PR-AUC": np.nanmean(pr_auc_list),
                }
            )

    return pd.DataFrame(results)


# ==============================================================================
# 3. Model Weight Extraction & Prediction Artifact
# ==============================================================================


def get_binary_logit_weights(
    df: pd.DataFrame,
    y_label: str,
    x_label: str,
    c: float,
    tau: float,
    class_weight: Optional[Union[str, dict]] = "balanced",
    random_state: int = 42,
    max_iter: int = 1000,
) -> Dict[str, Any]:
    """Fits Logistic Regression on raw embeddings without feature scaling,

    matching the final model in logistic_regression.py.
    """
    if c <= 0:
        raise ValueError("c must be strictly greater than 0.")

    X, y = _extract_x_y(df, y_label, x_label)

    clf = LogisticRegression(
        C=float(c),
        class_weight=class_weight,
        solver="lbfgs",
        max_iter=max_iter,
        random_state=random_state,
    )
    clf.fit(X, y)

    w = clf.coef_.ravel()
    b = float(clf.intercept_[0])

    def predict_with_tau(new_embeddings: np.ndarray) -> np.ndarray:
        z = np.dot(new_embeddings, w) + b
        probs = np.where(
            z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z))
        )
        return (probs >= tau).astype(int)

    return {
        "weights": w,
        "intercept": b,
        "tau": tau,
        "c": c,
        "model": clf,
        "predict": predict_with_tau,
    }


# ==============================================================================
# 4. Evaluation on Arbitrary Datasets
# ==============================================================================


def evaluate_model_artifacts(
    df: pd.DataFrame,
    y_label: str,
    x_label: str,
    model_artifacts: Dict[str, Any],
) -> pd.DataFrame:
    """Evaluates the fitted model artifacts on any evaluation DataFrame (train, val, or test)."""
    X, y_true = _extract_x_y(df, y_label, x_label)

    w = model_artifacts["weights"]
    b = model_artifacts["intercept"]
    tau = model_artifacts["tau"]
    c_val = model_artifacts["c"]

    z = np.dot(X, w) + b
    y_probs = np.where(
        z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z))
    )
    y_pred = (y_probs >= tau).astype(int)

    m = _compute_metrics(y_true, y_pred, y_probs)

    return pd.DataFrame(
        [
            {
                "C": c_val,
                "tau": tau,
                "Accuracy": m["accuracy"],
                "Specificity": m["specificity"],
                "Precision": m["precision"],
                "Recall": m["recall"],
                "F1-Score": m["f1"],
                "ROC-AUC": m["roc_auc"],
                "PR-AUC": m["pr_auc"],
            }
        ]
    )