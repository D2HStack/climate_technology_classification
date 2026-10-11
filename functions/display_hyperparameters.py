import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def display_hyperparameters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Plots 2D heatmaps (C vs tau) for each metric and outputs a summary table
    showing the optimal (C, tau) configuration split into two regimes:
      - tau <= 0.5 (Recall-oriented / Low threshold regime)
      - tau > 0.5  (Precision-oriented / High threshold regime)

    Parameters
    ----------
    df : pd.DataFrame
        Output DataFrame from evaluate_binary_logit_grid containing columns:
        ['C', 'tau', 'Accuracy', 'Specificity', 'Precision',
         'Recall', 'F1-Score', 'ROC-AUC', 'PR-AUC'].

    Returns
    -------
    pd.DataFrame
        Summary table comparing optimal configurations across both tau regimes.
    """
    metrics = [
        "Accuracy",
        "Specificity",
        "Precision",
        "Recall",
        "F1-Score",
        "ROC-AUC",
        "PR-AUC",
    ]
    available_metrics = [m for m in metrics if m in df.columns]

    # Partition into the two regimes (excluding trivial edge boundaries 0.0 and 1.0)
    df_low = df[(df["tau"] > 0.0) & (df["tau"] <= 0.5)]
    df_high = df[(df["tau"] > 0.5) & (df["tau"] < 1.0)]

    optimal_records = []

    for metric in available_metrics:
        if metric in ["ROC-AUC", "PR-AUC"]:
            # AUC metrics evaluate rank across all thresholds, independent of tau
            c_grouped = df.groupby("C")[metric].mean()
            best_c = c_grouped.idxmax()
            best_score = round(float(c_grouped.max()), 4)

            optimal_records.append(
                {
                    "Metric": metric,
                    "C (τ ≤ 0.5)": best_c,
                    "tau (τ ≤ 0.5)": "N/A (all τ)",
                    "Max Score (τ ≤ 0.5)": best_score,
                    "C (τ > 0.5)": best_c,
                    "tau (τ > 0.5)": "N/A (all τ)",
                    "Max Score (τ > 0.5)": best_score,
                }
            )
        else:
            # Regime 1: tau <= 0.5
            if not df_low.empty:
                best_low_row = df_low.loc[df_low[metric].idxmax()]
                c_low = best_low_row["C"]
                tau_low = best_low_row["tau"]
                score_low = round(float(best_low_row[metric]), 4)
            else:
                c_low, tau_low, score_low = np.nan, np.nan, np.nan

            # Regime 2: tau > 0.5
            if not df_high.empty:
                best_high_row = df_high.loc[df_high[metric].idxmax()]
                c_high = best_high_row["C"]
                tau_high = best_high_row["tau"]
                score_high = round(float(best_high_row[metric]), 4)
            else:
                c_high, tau_high, score_high = np.nan, np.nan, np.nan

            optimal_records.append(
                {
                    "Metric": metric,
                    "C (τ ≤ 0.5)": c_low,
                    "tau (τ ≤ 0.5)": tau_low,
                    "Max Score (τ ≤ 0.5)": score_low,
                    "C (τ > 0.5)": c_high,
                    "tau (τ > 0.5)": tau_high,
                    "Max Score (τ > 0.5)": score_high,
                }
            )

    best_params_df = pd.DataFrame(optimal_records)

    # --- Plot Heatmaps ---
    n_metrics = len(available_metrics)
    cols = 3
    rows = int(np.ceil(n_metrics / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(6.5 * cols, 5 * rows))
    axes = np.array(axes).flatten()

    for i, metric in enumerate(available_metrics):
        ax = axes[i]
        pivot_table = df.pivot(index="C", columns="tau", values=metric).sort_index(
            ascending=False
        )

        sns.heatmap(
            pivot_table,
            ax=ax,
            cmap="viridis",
            cbar=True,
            fmt=".2f",
            linewidths=0.2,
        )

        row_info = best_params_df[best_params_df["Metric"] == metric].iloc[0]
        if metric in ["ROC-AUC", "PR-AUC"]:
            title = f"{metric}\n(Max: {row_info['Max Score (τ ≤ 0.5)']} at C={row_info['C (τ ≤ 0.5)']})"
        else:
            title = (
                f"{metric}\n"
                f"τ ≤ 0.5: {row_info['Max Score (τ ≤ 0.5)']} (C={row_info['C (τ ≤ 0.5)']}, τ={row_info['tau (τ ≤ 0.5)']})\n"
                f"τ > 0.5: {row_info['Max Score (τ > 0.5)']} (C={row_info['C (τ > 0.5)']}, τ={row_info['tau (τ > 0.5)']})"
            )

        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel(r"Decision Threshold ($\tau$)", fontsize=10)
        ax.set_ylabel(r"Inverse Regularization ($C$)", fontsize=10)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

    # Display the formatted table
    print("\n" + "=" * 95)
    print("OPTIMAL HYPERPARAMETERS: LOW REGIME (τ ≤ 0.5) vs HIGH REGIME (τ > 0.5)")
    print("=" * 95)
    print(best_params_df.to_string(index=False))
    print("=" * 95)

    return best_params_df

############### Multi-Logit Hyperparameter Tuning Function ###############
from typing import Optional, Union
import numpy as np
import pandas as pd


def display_hyperparameters_multi(
    df_results: pd.DataFrame,
    metric: str = "F1-Score",
) -> pd.DataFrame:
    """Computes the optimal (C, tau) configurations for multi-label grid results,

    comparing the low regime (tau <= 0.5) against the high regime (tau > 0.5).

    Parameters
    ----------
    df_results : pd.DataFrame
        Output DataFrame from evaluate_multi_logit_grid containing columns:
        ['y_label', 'C', 'tau', 'Accuracy', 'Specificity', 'Precision',
         'Recall', 'F1-Score', 'ROC-AUC', 'PR-AUC'].
    metric : str, default "F1-Score"
        The specific metric to summarize with one row per y_label.
        Available: 'F1-Score', 'Precision', 'Recall', 'Accuracy', 'Specificity',
                   'ROC-AUC', 'PR-AUC', or 'all'.

    Returns
    -------
    pd.DataFrame
        Table summarizing optimal C, tau, and scores across both tau regimes.
    """
    all_metrics = [
        "Accuracy",
        "Specificity",
        "Precision",
        "Recall",
        "F1-Score",
        "ROC-AUC",
        "PR-AUC",
    ]

    metrics_to_eval = (
        all_metrics
        if metric.lower() == "all"
        else [m for m in all_metrics if m.lower() == metric.lower()]
    )
    if not metrics_to_eval:
        raise ValueError(
            f"Metric '{metric}' not recognized. Choose from: {all_metrics} or 'all'."
        )

    # Exclude boundary extremes tau = 0.0 and tau = 1.0 for threshold-dependent metrics
    df_low = df_results[
        (df_results["tau"] > 0.0) & (df_results["tau"] <= 0.5)
    ].copy()
    df_high = df_results[
        (df_results["tau"] > 0.5) & (df_results["tau"] < 1.0)
    ].copy()

    records = []
    labels = df_results["y_label"].unique()

    for target in labels:
        df_target = df_results[df_results["y_label"] == target]
        df_target_low = df_low[df_low["y_label"] == target]
        df_target_high = df_high[df_high["y_label"] == target]

        for m in metrics_to_eval:
            if m in ["ROC-AUC", "PR-AUC"]:
                # AUC metrics are rank-based and invariant to tau; compute best C across all tau
                c_grouped = df_target.groupby("C")[m].mean()
                best_c = c_grouped.idxmax()
                best_score = round(float(c_grouped.max()), 4)

                c_low, tau_low, score_low = best_c, "N/A (all τ)", best_score
                c_high, tau_high, score_high = best_c, "N/A (all τ)", best_score
            else:
                # Regime 1: tau <= 0.5
                if not df_target_low.empty:
                    best_low_row = df_target_low.loc[
                        df_target_low[m].idxmax()
                    ]
                    c_low = best_low_row["C"]
                    tau_low = best_low_row["tau"]
                    score_low = round(float(best_low_row[m]), 4)
                else:
                    c_low, tau_low, score_low = np.nan, np.nan, np.nan

                # Regime 2: tau > 0.5
                if not df_target_high.empty:
                    best_high_row = df_target_high.loc[
                        df_target_high[m].idxmax()
                    ]
                    c_high = best_high_row["C"]
                    tau_high = best_high_row["tau"]
                    score_high = round(float(best_high_row[m]), 4)
                else:
                    c_high, tau_high, score_high = np.nan, np.nan, np.nan

            rec = {
                "y_label": target,
                "Metric": m,
                "C (τ ≤ 0.5)": c_low,
                "tau (τ ≤ 0.5)": tau_low,
                "Max Score (τ ≤ 0.5)": score_low,
                "C (τ > 0.5)": c_high,
                "tau (τ > 0.5)": tau_high,
                "Max Score (τ > 0.5)": score_high,
            }
            records.append(rec)

    summary_df = pd.DataFrame(records)

    # Format header display
    title_metric = metric.upper() if metric.lower() != "all" else "ALL METRICS"
    print("\n" + "=" * 98)
    print(
        f"OPTIMAL HYPERPARAMETERS: LOW REGIME (τ ≤ 0.5) vs HIGH REGIME (τ > 0.5) FOR {title_metric}"
    )
    print("=" * 98)

    # If single metric, drop 'Metric' column so table is 1 clean row per y_label
    if metric.lower() != "all":
        display_df = summary_df.drop(columns=["Metric"])
        print(display_df.to_string(index=False))
        print("=" * 98)
        return display_df
    else:
        print(summary_df.to_string(index=False))
        print("=" * 98)
        return summary_df