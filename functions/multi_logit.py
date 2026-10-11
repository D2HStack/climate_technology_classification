"""multi_logit.py.

Multi-target Logistic Regression pipeline for document embeddings across
overlapping taxonomy categories. Supports independent regularized logit heads,
per-label thresholding, dual-consensus conjunctive evaluation, and grid search.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import ast
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
# 1. Feature Extraction & Robust Metric Helpers
# ==============================================================================


def _extract_x_only(
    new_x: Union[np.ndarray, pd.Series, list],
) -> np.ndarray:
    """Extracts and L2-normalizes input embeddings across Series, ndarray, or list of vectors."""
    # Convert Series to list/array
    if isinstance(new_x, pd.Series):
        new_x = new_x.values

    # If it's already a clean 2D float ndarray, normalize directly
    if isinstance(new_x, np.ndarray) and new_x.ndim == 2 and np.issubdtype(new_x.dtype, np.floating):
        return normalize(new_x, norm="l2")

    # If it's a sequence/1D container of vectors
    first_elem = new_x[0] if len(new_x) > 0 else []

    if isinstance(first_elem, str):
        X = np.vstack([ast.literal_eval(v) for v in new_x]).astype(np.float64)
    else:
        X = np.vstack(list(new_x)).astype(np.float64)

    return normalize(X, norm="l2")


def _extract_x_y(
    df: pd.DataFrame, y_label: str, x_label: str
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Extracts L2-normalized embeddings X and binary label array y."""
    clean_df = df.dropna(subset=[x_label]).copy()
    raw_emb = clean_df[x_label].values

    first_elem = raw_emb[0] if len(raw_emb) > 0 else []
    if isinstance(first_elem, str):
        X = np.vstack([ast.literal_eval(v) for v in raw_emb]).astype(np.float64)
    else:
        X = np.vstack(list(raw_emb)).astype(np.float64)

    X = normalize(X, norm="l2")
    y = clean_df[y_label].fillna(False).astype(int).to_numpy()
    return X, y, clean_df


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
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(roc_val, 4) if pd.notna(roc_val) else np.nan,
        "pr_auc": round(pr_val, 4) if pd.notna(pr_val) else np.nan,
    }


# ==============================================================================
# 2. Multi-Label Grid Search
# ==============================================================================


def evaluate_multi_logit_grid(
    df: pd.DataFrame,
    y_labels: List[str],
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
    """Evaluates Logistic Regression over a grid of C and tau for multiple

    overlapping targets using L2-normalized embeddings.
    """
    if c_min <= 0:
        raise ValueError("c_min must be strictly greater than 0.")

    eps = 1e-9
    c_values = np.arange(c_min, c_max + eps, c_increment)
    tau_values = np.arange(tau_min, tau_max + eps, tau_increment)

    all_results = []

    for target in y_labels:
        if target not in df.columns:
            continue

        X, y, _ = _extract_x_y(df, target, x_label)
        n_pos = int(y.sum())

        if n_pos < 2:
            print(
                f"Skipping '{target}': Insufficient positive samples ({n_pos}) for stratified grid search."
            )
            continue

        effective_splits = min(n_rep, n_pos)
        sss = StratifiedShuffleSplit(
            n_splits=effective_splits, test_size=split, random_state=random_state
        )

        try:
            splits = list(sss.split(X, y))
        except ValueError as e:
            print(f"Skipping '{target}': {e}")
            continue

        for c_val in c_values:
            rep_data = []

            for train_idx, test_idx in splits:
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

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

                all_results.append(
                    {
                        "y_label": target,
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

    return pd.DataFrame(all_results)


# ==============================================================================
# 3. Model Weight Fitting & Multi-Logit Artifacts
# ==============================================================================


def get_multi_logit_weights(
    df: pd.DataFrame,
    y_labels: List[str],
    y_global_label: str,
    x_label: str,
    hyperparameters: pd.DataFrame,
    class_weight: Optional[Union[str, dict]] = "balanced",
    random_state: int = 42,
    max_iter: int = 1000,
) -> Dict[str, Any]:
    """Fits independent logistic regressions on L2-normalized embeddings for each

    constituent category and the global umbrella target.
    """
    hp_map = hyperparameters.set_index("label")
    req_cols = {"c_param", "tau"}
    if not req_cols.issubset(hp_map.columns):
        raise ValueError(f"hyperparameters DataFrame must contain columns: {req_cols}")

    constituent_labels = [lbl for lbl in y_labels if lbl != y_global_label]
    all_targets = list(dict.fromkeys(constituent_labels + [y_global_label]))

    missing = [lbl for lbl in all_targets if lbl not in hp_map.index]
    if missing:
        raise KeyError(f"Missing hyperparameters for labels: {missing}")

    clean_df = df.dropna(subset=[x_label]).copy()
    raw_emb = clean_df[x_label].values
    if isinstance(raw_emb[0], str):
        X = np.vstack([ast.literal_eval(v) for v in raw_emb]).astype(np.float64)
    else:
        X = np.vstack(raw_emb).astype(np.float64)
    X = normalize(X, norm="l2")

    models = {}
    weights_dict = {}
    intercepts_dict = {}
    taus_dict = {}
    c_dict = {}

    for target in all_targets:
        c_val = float(hp_map.loc[target, "c_param"])
        tau_val = float(hp_map.loc[target, "tau"])
        y_target = clean_df[target].fillna(False).astype(int).to_numpy()

        clf = LogisticRegression(
            C=c_val,
            class_weight=class_weight,
            solver="lbfgs",
            max_iter=max_iter,
            random_state=random_state,
        )

        if y_target.sum() == 0:
            models[target] = None
            weights_dict[target] = np.zeros(X.shape[1])
            intercepts_dict[target] = -10.0
        else:
            clf.fit(X, y_target)
            models[target] = clf
            weights_dict[target] = clf.coef_.ravel()
            intercepts_dict[target] = float(clf.intercept_[0])

        taus_dict[target] = tau_val
        c_dict[target] = c_val

    df_weights = pd.DataFrame(weights_dict)
    s_intercepts = pd.Series(intercepts_dict)
    s_taus = pd.Series(taus_dict)
    s_c = pd.Series(c_dict)

    def predict_proba(
        new_embeddings: Union[np.ndarray, pd.Series, list],
    ) -> pd.DataFrame:
        X_mat = _extract_x_only(new_embeddings)
        probs_dict = {}
        for target in all_targets:
            clf = models[target]
            if clf is not None:
                probs_dict[target] = clf.predict_proba(X_mat)[:, 1]
            else:
                probs_dict[target] = np.zeros(len(X_mat))
        return pd.DataFrame(probs_dict)

    def predict_detailed(
        new_embeddings: Union[np.ndarray, pd.Series, list],
    ) -> Dict[str, Any]:
        df_probs = predict_proba(new_embeddings)

        preds_constituent = pd.DataFrame(
            {
                col: (df_probs[col].values >= s_taus[col]).astype(int)
                for col in constituent_labels
            }
        )

        tau_g = s_taus[y_global_label]
        pred_global = (df_probs[y_global_label].values >= tau_g).astype(int)

        any_constituent = preds_constituent.any(axis=1).astype(int).to_numpy()
        joint_pred = (any_constituent & pred_global).astype(int)

        return {
            "probabilities": df_probs,
            "constituent_preds": preds_constituent,
            "global_pred": pred_global,
            "any_constituent_pred": any_constituent,
            "joint_pred": joint_pred,
        }

    def predict(
        new_embeddings: Union[np.ndarray, pd.Series, list],
    ) -> np.ndarray:
        return predict_detailed(new_embeddings)["joint_pred"]

    return {
        "weight_matrix": df_weights,
        "weights": df_weights,
        "weights_raw": df_weights,
        "intercepts": s_intercepts,
        "taus": s_taus,
        "c_params": s_c,
        "models": models,
        "constituent_labels": constituent_labels,
        "global_label": y_global_label,
        "predict_proba": predict_proba,
        "predict_detailed": predict_detailed,
        "predict": predict,
    }


# ==============================================================================
# 4. Comprehensive Artifacts Evaluation
# ==============================================================================


def evaluate_multi_model_artifacts(
    df: pd.DataFrame,
    y_global_label: str,
    x_label: str,
    multi_artifacts: Dict[str, Any],
) -> pd.DataFrame:
    """Evaluates multi-model artifacts using normalized embeddings and correct

    inclusion-exclusion joint probability aggregation.
    """
    constituent_labels = multi_artifacts["constituent_labels"]
    s_taus = multi_artifacts["taus"]
    s_c = multi_artifacts["c_params"]

    if y_global_label not in df.columns:
        raise KeyError(f"Global target '{y_global_label}' not found in DataFrame.")

    clean_df = df.dropna(subset=[x_label]).copy()
    X = _extract_x_only(clean_df[x_label].values)

    detailed = multi_artifacts["predict_detailed"](X)
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
    avail_constituents = [c for c in constituent_labels if c in clean_df.columns]
    y_true_any_constituent = (
        clean_df[avail_constituents].fillna(False).astype(bool).any(axis=1)
    )
    y_true_joint = (
        (clean_df[y_global_label].fillna(False).astype(bool) & y_true_any_constituent)
        .astype(int)
        .to_numpy()
    )

    # Inclusion-exclusion continuous union: P(Any) = 1 - Prod(1 - P_k)
    P_const_mat = df_probs[avail_constituents].to_numpy()
    p_any_constituent = 1.0 - np.prod(1.0 - P_const_mat, axis=1)

    # Joint continuous ranking score
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

############## PAIRWISE DYADIC EVALUATION ##############

import ast
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize


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

def evaluate_pairwise_model_artifacts(
    df_ventures: pd.DataFrame,
    df_anchors: pd.DataFrame,
    col_emb: str,
    df_anchors_emb: str,
    df_anchors_label: str,
    categories: List[str],
    y_global_label: str,
    multi_artifacts: Dict[str, Any],
) -> pd.DataFrame:
    """Evaluates multi-logit models on their matching (venture, anchor) dyads only,

    producing true venture-level (N) metrics instead of N*K inflated metrics.
    """
    clean_v = df_ventures.dropna(subset=[col_emb]).copy().reset_index(drop=True)
    clean_a = (
        df_anchors.dropna(subset=[df_anchors_emb, df_anchors_label])
        .copy()
        .set_index(df_anchors_label)
    )

    N = len(clean_v)
    K = len(categories)

    # 1. Extract and normalize
    X = _unpack_and_normalize_embeddings(clean_v[col_emb])
    A_dict = {
        cat: _unpack_and_normalize_embeddings(
            pd.Series([clean_a.loc[cat, df_anchors_emb]])
        )[0]
        for cat in categories
        if cat in clean_a.index
    }

    models = multi_artifacts["models"]
    s_taus = multi_artifacts["taus"]
    s_c = multi_artifacts["c_params"]

    # 2. Compute probabilities ONLY for matching (i, k) pairs
    # P_mat shape: (N, K)
    P_mat = np.zeros((N, K), dtype=np.float64)

    for k_idx, cat in enumerate(categories):
        if cat not in models or models[cat] is None:
            continue
        a_k = A_dict[cat]

        # Compute z_ik for all N ventures paired with anchor a_k
        hadamard = X * a_k  # (N, D)
        manhattan = np.abs(X - a_k)  # (N, D)
        cosine = np.sum(hadamard, axis=1, keepdims=True)  # (N, 1)
        Z_k = np.hstack([hadamard, manhattan, cosine])  # (N, 2D+1)
        Z_k = normalize(Z_k, norm="l2")

        clf = models[cat]
        P_mat[:, k_idx] = clf.predict_proba(Z_k)[:, 1]

    # Convert to DataFrame of venture-level probabilities
    df_probs = pd.DataFrame(P_mat, columns=categories)

    # Global screener probability (evaluated on raw or macro-aggregate)
    if y_global_label in models and models[y_global_label] is not None:
        # If is_climate was trained on a specific anchor or raw emb
        clf_g = models[y_global_label]
        # P_global evaluated on constituent union max or trained head
        p_global = P_mat.max(axis=1)  # Or clf_g prediction
    else:
        p_global = 1.0 - np.prod(1.0 - P_mat, axis=1)

    tau_global = s_taus.get(y_global_label, 0.50)
    pred_global = (p_global >= tau_global).astype(int)

    records = []

    # 3. Constituent Categories
    for k_idx, cat in enumerate(categories):
        y_true = clean_v[cat].fillna(False).astype(int).to_numpy()
        tau_val = s_taus[cat]
        y_prob = P_mat[:, k_idx]
        y_pred = (y_prob >= tau_val).astype(int)

        m = _compute_metrics(y_true, y_pred, y_prob)
        records.append(
            {
                "Target / Rule": cat,
                "Type": "Constituent",
                "C": s_c[cat],
                "tau": tau_val,
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

    # 4. Joint Rule
    y_true_joint = (
        clean_v[categories].fillna(False).astype(bool).any(axis=1).astype(int)
    )
    y_pred_constituents = np.array(
        [(P_mat[:, i] >= s_taus[cat]).astype(int) for i, cat in enumerate(categories)]
    ).T
    pred_any = y_pred_constituents.any(axis=1).astype(int)
    joint_pred = (pred_any & pred_global).astype(int)
    p_any = 1.0 - np.prod(1.0 - P_mat, axis=1)
    joint_prob = p_global * p_any

    m_joint = _compute_metrics(y_true_joint.to_numpy(), joint_pred, joint_prob)
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