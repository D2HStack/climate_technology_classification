"""dyadic_features.py

Vectorized construction of the (2D + 1) pairwise interaction space:
    z_ik = [ x_i (*) a_k, |x_i - a_k|, x_i^T a_k ]
"""

from typing import Optional, Sequence, Union
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


def build_pairwise_embeddings_df(
    df: pd.DataFrame,
    df_emb: str,
    df_anchors: pd.DataFrame,
    df_anchors_emb: str,
    df_anchors_label: str,
    target_categories: Optional[Sequence[str]] = None,
    output_emb_col: str = "pairwise_emb",
    output_label_col: str = "anchor_label",
    global_target_col: str = "is_climate",
) -> pd.DataFrame:
    """Couples every venture disclosure embedding x_i with candidate anchors a_k

    to produce a dyadic dataset where binary category flags reflect the TRUE match
    for that specific paired anchor.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset of ventures (length N).
    df_emb : str
        Column name holding venture document embeddings x_i (dimension D).
    df_anchors : pd.DataFrame
        Dataset of anchors (length K).
    df_anchors_emb : str
        Column name in `df_anchors` holding anchor embeddings a_k (dimension D).
    df_anchors_label : str
        Column name in `df_anchors` holding the anchor category name (e.g., 'is_energy').
    target_categories : Sequence[str], optional
        List of category column names to mask per dyad. If None, uses all values in df_anchors[df_anchors_label].
    output_emb_col : str, default 'pairwise_emb'
        Name of output column containing z_ik interaction vectors (dimension 2D + 1).
    output_label_col : str, default 'anchor_label'
        Name of output column containing the anchor category identifier.
    global_target_col : str, default 'is_climate'
        Name of the umbrella column.

    Returns
    -------
    pd.DataFrame
        DataFrame of shape (N * K, ...) where each row (i, k) has:
          - `anchor_label`: category k
          - `pairwise_emb`: interaction vector z_ik
          - category k column: 1 IF venture i has category k, ELSE 0
          - all other category columns != k: 0
          - `is_match`: 1 if venture i belongs to category k, else 0
    """
    clean_ventures = df.dropna(subset=[df_emb]).copy().reset_index(drop=True)
    clean_anchors = (
        df_anchors.dropna(subset=[df_anchors_emb, df_anchors_label])
        .copy()
        .reset_index(drop=True)
    )

    N = len(clean_ventures)
    K = len(clean_anchors)

    if N == 0 or K == 0:
        raise ValueError("Input dataframe or anchors dataframe is empty.")

    # 1. Normalize embeddings: X in R^(N x D), A in R^(K x D)
    X = _unpack_and_normalize_embeddings(clean_ventures[df_emb])
    A = _unpack_and_normalize_embeddings(clean_anchors[df_anchors_emb])

    if X.shape[1] != A.shape[1]:
        raise ValueError(
            f"Embedding dim mismatch: ventures D={X.shape[1]}, anchors D={A.shape[1]}"
        )
    D = X.shape[1]

    # 2. Vectorized 3D Dyadic Broadcasting: (N, K, D)
    X_broad = np.repeat(X[:, np.newaxis, :], K, axis=1)
    A_broad = np.repeat(A[np.newaxis, :, :], N, axis=0)

    hadamard = X_broad * A_broad
    manhattan = np.abs(X_broad - A_broad)
    cosine = np.sum(hadamard, axis=-1, keepdims=True)

    Z_tensor = np.concatenate([hadamard, manhattan, cosine], axis=-1)
    Z_flat = Z_tensor.reshape(N * K, 2 * D + 1)

    # 3. Expand DataFrame to N * K rows
    df_expanded = clean_ventures.loc[
        clean_ventures.index.repeat(K)
    ].reset_index(drop=True)

    # 4. Attach anchor labels (tiled per venture)
    anchor_labels = clean_anchors[df_anchors_label].values
    tiled_anchors = np.tile(anchor_labels, N)
    df_expanded[output_label_col] = tiled_anchors
    df_expanded[output_emb_col] = list(Z_flat)

    # 5. Mask category columns to reflect ONLY the paired anchor
    if target_categories is None:
        target_categories = [
            c for c in anchor_labels if c in clean_ventures.columns
        ]

    # Compute single dyadic match flag
    is_match_list = []
    for cat in target_categories:
        if cat in df_expanded.columns:
            # Mask: Positive ONLY if the row's anchor_label matches AND venture has this label
            mask_this_anchor = df_expanded[output_label_col] == cat
            df_expanded[cat] = (
                df_expanded[cat].fillna(False).astype(bool) & mask_this_anchor
            ).astype(int)

    # Add single binary target column for unified pairwise matching
    df_expanded["is_match"] = (
        df_expanded[target_categories].any(axis=1).astype(int)
    )

    # Preserve global umbrella flag: true if the specific paired dyad is positive
    if global_target_col in df_expanded.columns:
        df_expanded[global_target_col] = df_expanded["is_match"]

    return df_expanded