"""augmented_logit.py.

Augmented Dyadic Logistic Regression pipeline for document embeddings and anchor
representations. Implements the Unified Pairwise Matcher (z_ik in R^(2D+1)) with
category-specific decision thresholds (tau_k) and dual-consensus global gating.
"""

import ast
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import normalize

# ==============================================================================
# 1. Feature Extraction & Dyadic Geometric Coupling
# ==============================================================================


def _unpack_and_normalize_embeddings(
    series_or_array: Union[pd.Series, np.ndarray, list],
) -> np.ndarray:
    """Unpacks string-serialized or list/array embeddings into an L2-normalized float64 2D matrix."""
    if isinstance(series_or_array, pd.Series):
        values = series_or_array.values
    else:
        values = series_or_array

    first_elem = values[0] if len(values) > 0 else []
    if isinstance(first_elem, str):
        X = np.vstack([ast.literal_eval(v) for v in values]).astype(np.float64)
    else:
        X = np.vstack(list(values)).astype(np.float64)

    return normalize(X, norm="l2")


def _compute_dyadic_features(
    X: np.ndarray,
    A: np.ndarray,
) -> np.ndarray:
    """Computes the (2D + 1) interaction tensor: [X (*) A, |X - A|, X^T A].

    Parameters
    ----------
    X : np.ndarray, shape (N, D)
    A : np.ndarray, shape (K, D) or (N, D) for pre-aligned pairs

    Returns
    -------
    Z : np.ndarray, shape (N, 2D + 1) or (N*K, 2D + 1)
    """
    if X.ndim == 2 and A.ndim == 2 and X.shape[0] == A.shape[0]:
        # Fast path: pre-aligned 1-to-1 rows
        hadamard = X * A
        manhattan = np.abs(X - A)
        cosine = np.sum(hadamard, axis=1, keepdims=True)
        return np.hstack([hadamard, manhattan, cosine])

    N, D = X.shape
    K, D_a = A.shape
    assert D == D_a, f"Dimension mismatch: X has {D}, A has {D_a}"

    X_broad = np.repeat(X[:, np.newaxis, :], K, axis=1)  # (N, K, D)
    A_broad = np.repeat(A[np.newaxis, :, :], N, axis=0)  # (N, K, D)

    hadamard = X_broad * A_broad
    manhattan = np.abs(X_broad - A_broad)
    cosine = np.sum(hadamard, axis=-1, keepdims=True)

    Z_tensor = np.concatenate([hadamard, manhattan, cosine], axis=-1)
    return Z_tensor.reshape(N * K, 2 * D + 1)


def _compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray
) -> Dict[str, float]:
    """Computes binary classification metrics handling edge cases where support is 0 or 1."""
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    if n_pos > 0 and n_neg > 0:
        try:
            roc_val = float(roc_auc_score(y_true, y_proba))
        except (ValueError, IndexError):
            roc_val = np.nan

        try:
            prec_arr, rec_arr, _ = precision_recall_curve(y_true, y_proba)
            pr_val = float(auc(rec_arr, prec_arr))
        except (ValueError, IndexError):
            pr_val = np.nan
    else:
        roc_val = np.nan
        pr_val = np.nan

    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "specificity": round(spec, 4),
        "precision": round(
            float(precision_score(y_true, y_pred, zero_division=0)), 4
        ),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(roc_val, 4) if pd.notna(roc_val) else np.nan,
        "pr_auc": round(pr_val, 4) if pd.notna(pr_val) else np.nan,
    }


# ==============================================================================
# 2. Augmented Model Weight Fitting
# ==============================================================================


def get_augmented_logit_weights(
    df: pd.DataFrame,
    df_anchors: pd.DataFrame,
    y_labels: List[str],
    y_global_label: str,
    x_label: str = "snapshot_emb",
    anchor_emb_col: str = "embedding",
    anchor_label_col: str = "label",
    hyperparameters: Optional[pd.DataFrame] = None,
    C: float = 1.0,
    class_weight: Optional[Union[str, dict]] = "balanced",
    global_class_weight: Optional[Union[str, dict]] = None,
    random_state: int = 42,
    max_iter: int = 1000,
) -> Dict[str, Any]:
    """Fits a Unified Pairwise Relational Classifier across venture-anchor dyads (z_ik in R^(2D+1))
    plus a high-precision binary screener on the global umbrella target.
    """
    clean_df = df.dropna(subset=[x_label]).copy().reset_index(drop=True)
    clean_anchors = (
        df_anchors.dropna(subset=[anchor_emb_col, anchor_label_col])
        .copy()
        .set_index(anchor_label_col)
    )

    constituent_labels = [c for c in y_labels if c != y_global_label]
    K = len(constituent_labels)
    N = len(clean_df)

    # 1. Normalize embeddings: X in R^(N x D), A in R^(K x D)
    X = _unpack_and_normalize_embeddings(clean_df[x_label])
    A_list = [
        _unpack_and_normalize_embeddings(
            pd.Series([clean_anchors.loc[cat, anchor_emb_col]])
        )[0]
        for cat in constituent_labels
    ]
    A = np.vstack(A_list)

    # 2. Build Dyadic Interaction Matrix Z (N * K, 2D + 1)
    Z = _compute_dyadic_features(X, A)

    # Ground truth matrix Y: (N, K) -> flatten to (N * K)
    Y_mat = np.zeros((N, K), dtype=int)
    for k_idx, cat in enumerate(constituent_labels):
        if cat in clean_df.columns:
            Y_mat[:, k_idx] = clean_df[cat].fillna(False).astype(int).to_numpy()

    y_dyadic = Y_mat.ravel()

    # 3. Fit Unified Pairwise Matcher (balanced for sensitivity across rare classes)
    unified_clf = LogisticRegression(
        C=C,
        class_weight=class_weight,
        solver="lbfgs",
        max_iter=max_iter,
        random_state=random_state,
    )
    unified_clf.fit(Z, y_dyadic)

    # 4. Fit Global Umbrella Screener (unweighted for high precision)
    y_global = (
        clean_df[y_global_label].fillna(False).astype(int).to_numpy()
        if y_global_label in clean_df.columns
        else (Y_mat.sum(axis=1) > 0).astype(int)
    )

    global_clf = LogisticRegression(
        C=C,
        class_weight=global_class_weight,
        solver="lbfgs",
        max_iter=max_iter,
        random_state=random_state,
    )
    global_clf.fit(X, y_global)

    # 5. Extract thresholds from hyperparameters DataFrame
    s_taus = {}
    s_c = {}
    if hyperparameters is not None:
        hp_map = hyperparameters.set_index("label")
        for cat in constituent_labels + [y_global_label]:
            s_taus[cat] = float(hp_map.loc[cat, "tau"]) if cat in hp_map.index else 0.50
            s_c[cat] = float(hp_map.loc[cat, "c_param"]) if cat in hp_map.index else C
    else:
        for cat in constituent_labels + [y_global_label]:
            s_taus[cat] = 0.50
            s_c[cat] = C

    s_taus = pd.Series(s_taus)
    s_c = pd.Series(s_c)

    def predict_detailed(
        new_embeddings: Union[np.ndarray, pd.Series, list],
    ) -> Dict[str, Any]:
        X_eval = _unpack_and_normalize_embeddings(new_embeddings)
        N_eval = len(X_eval)

        # Score across K dyads
        Z_eval = _compute_dyadic_features(X_eval, A)
        probs_dyadic = unified_clf.predict_proba(Z_eval)[:, 1]
        P_mat = probs_dyadic.reshape(N_eval, K)

        df_probs = pd.DataFrame(P_mat, columns=constituent_labels)
        df_probs[y_global_label] = global_clf.predict_proba(X_eval)[:, 1]

        preds_constituent = pd.DataFrame(
            {
                col: (df_probs[col].values >= s_taus[col]).astype(int)
                for col in constituent_labels
            }
        )

        pred_global = (
            df_probs[y_global_label].values >= s_taus[y_global_label]
        ).astype(int)
        any_constituent = preds_constituent.any(axis=1).astype(int).to_numpy()
        joint_pred = (any_constituent & pred_global).astype(int)

        return {
            "probabilities": df_probs,
            "constituent_preds": preds_constituent,
            "global_pred": pred_global,
            "any_constituent_pred": any_constituent,
            "joint_pred": joint_pred,
        }

    return {
        "unified_model": unified_clf,
        "global_model": global_clf,
        "weights": unified_clf.coef_.ravel(),
        "weight_matrix": unified_clf.coef_.T,
        "intercept": float(unified_clf.intercept_[0]),
        "intercepts": pd.Series({
            "unified_pairwise": float(unified_clf.intercept_[0]),
            y_global_label: float(global_clf.intercept_[0]),
        }),
        "taus": s_taus,
        "c_params": s_c,
        "constituent_labels": constituent_labels,
        "global_label": y_global_label,
        "anchors_matrix": A,
        "predict_detailed": predict_detailed,
    }

# ==============================================================================
# 3. Model Artifacts Evaluation (Venture-Level N, No Row Expansion)
# ==============================================================================


def evaluate_augmented_model_artifacts(
    df: pd.DataFrame,
    y_global_label: str,
    x_label: str,
    augmented_artifacts: Dict[str, Any],
) -> pd.DataFrame:
    """Evaluates the Unified Pairwise Matcher at the unexpanded venture level (N),

    preserving true Support and preventing false alarm inflation.
    """
    constituent_labels = augmented_artifacts["constituent_labels"]
    s_taus = augmented_artifacts["taus"]
    s_c = augmented_artifacts["c_params"]

    clean_df = df.dropna(subset=[x_label]).copy().reset_index(drop=True)
    X = _unpack_and_normalize_embeddings(clean_df[x_label])

    detailed = augmented_artifacts["predict_detailed"](X)
    df_probs = detailed["probabilities"]
    df_preds_constituent = detailed["constituent_preds"]

    tau_global = s_taus[y_global_label]
    prob_global = df_probs[y_global_label].to_numpy()
    pred_global = (prob_global >= tau_global).astype(int)
    joint_pred = detailed["joint_pred"]

    records = []

    # 1. Constituent Categories
    for col in constituent_labels:
        if col not in clean_df.columns:
            continue

        y_true = clean_df[col].fillna(False).astype(int).to_numpy()
        y_pred = df_preds_constituent[col].to_numpy()
        y_prob = df_probs[col].to_numpy()

        m = _compute_metrics(y_true, y_pred, y_prob)
        records.append(
            {
                "Target / Rule": col,
                "Type": "Constituent",
                "C": s_c[col],
                "tau": s_taus[col],
                "Support (Pos)": int(y_true.sum()),
                "Predicted (Pos)": int(y_pred.sum()),
                "Accuracy": m["accuracy"],
                "Specificity": m["specificity"],
                "Precision": m["precision"],
                "Recall": m["recall"],
                "F1-Score": m["f1"],
                "ROC-AUC": m["roc_auc"],
                "PR-AUC": m["pr_auc"],
            }
        )

    # 2. Global Umbrella Target
    y_true_global = clean_df[y_global_label].fillna(False).astype(int).to_numpy()
    m_global = _compute_metrics(y_true_global, pred_global, prob_global)
    records.append(
        {
            "Target / Rule": y_global_label,
            "Type": "Global Umbrella",
            "C": s_c[y_global_label],
            "tau": tau_global,
            "Support (Pos)": int(y_true_global.sum()),
            "Predicted (Pos)": int(pred_global.sum()),
            "Accuracy": m_global["accuracy"],
            "Specificity": m_global["specificity"],
            "Precision": m_global["precision"],
            "Recall": m_global["recall"],
            "F1-Score": m_global["f1"],
            "ROC-AUC": m_global["roc_auc"],
            "PR-AUC": m_global["pr_auc"],
        }
    )

    # 3. Dual-Consensus Joint Rule (Global ∩ Any Category)
    avail_constituents = [
        c for c in constituent_labels if c in clean_df.columns
    ]
    y_true_any_constituent = (
        clean_df[avail_constituents].fillna(False).astype(bool).any(axis=1)
    )
    y_true_joint = (
        (clean_df[y_global_label].fillna(False).astype(bool) & y_true_any_constituent)
        .astype(int)
        .to_numpy()
    )

    P_const_mat = df_probs[avail_constituents].to_numpy()
    p_any_constituent = 1.0 - np.prod(1.0 - P_const_mat, axis=1)
    joint_prob = prob_global * p_any_constituent

    m_joint = _compute_metrics(y_true_joint, joint_pred, joint_prob)
    records.append(
        {
            "Target / Rule": f"JOINT ({y_global_label} ∩ Any Category)",
            "Type": "Compound Rule",
            "C": "Multi",
            "tau": "Multi",
            "Support (Pos)": int(y_true_joint.sum()),
            "Predicted (Pos)": int(joint_pred.sum()),
            "Accuracy": m_joint["accuracy"],
            "Specificity": m_joint["specificity"],
            "Precision": m_joint["precision"],
            "Recall": m_joint["recall"],
            "F1-Score": m_joint["f1"],
            "ROC-AUC": m_joint["roc_auc"],
            "PR-AUC": m_joint["pr_auc"],
        }
    )

    return pd.DataFrame(records)