"""
Reusable plotting functions for anonymization experiment results.

Expected minimum input for accuracy-by-level:
    model, anonymization_level, accuracy

Expected minimum input for accuracy-by-k:
    model, k, accuracy

The current baseline-versus-Level-1 comparison CSV may instead contain wide columns
such as baseline_accuracy and level1_accuracy. Use plot_baseline_vs_level1() for that
format. Do not infer missing Medium/High levels or k values.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def _read_results(csv_path):
    """Load a results CSV and return its DataFrame."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Results CSV not found: {path}")
    return pd.read_csv(path)


def _require_columns(df, required, plot_name):
    missing = [column for column in required if column not in df.columns]
    if missing:
        available = ", ".join(map(str, df.columns))
        raise ValueError(
            f"{plot_name} needs columns {required}; missing {missing}. "
            f"Available columns: {available}"
        )


def plot_accuracy_by_level(csv_path, output_path=None):
    """
    Plot accuracy by anonymization level for each model.

    CSV columns required: model, anonymization_level, accuracy.
    Accuracy may be stored as a fraction (0-1) or percentage (0-100).
    """
    df = _read_results(csv_path)
    _require_columns(df, ["model", "anonymization_level", "accuracy"],
                     "plot_accuracy_by_level")
    df = df.copy()
    df["accuracy"] = pd.to_numeric(df["accuracy"], errors="raise")
    if not df.empty and df["accuracy"].max() <= 1:
        df["accuracy"] *= 100

    pivot = df.pivot_table(
        index="anonymization_level", columns="model", values="accuracy",
        aggfunc="mean"
    )
    ax = pivot.plot(kind="bar", figsize=(9, 5), edgecolor="black")
    ax.set_title("Accuracy by Anonymization Level")
    ax.set_xlabel("Anonymization Level")
    ax.set_ylabel("Accuracy (%)")
    ax.grid(axis="y", linestyle=":", alpha=0.7)
    ax.set_axisbelow(True)
    ax.legend(title="Model", frameon=False)
    fig = ax.get_figure()
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
    return ax


def plot_accuracy_by_k(csv_path, output_path=None):
    """
    Plot accuracy against k for each model.

    CSV columns required: model, k, accuracy.
    This function only plots k values present in the input CSV.
    """
    df = _read_results(csv_path)
    _require_columns(df, ["model", "k", "accuracy"], "plot_accuracy_by_k")
    df = df.copy()
    df["k"] = pd.to_numeric(df["k"], errors="raise")
    df["accuracy"] = pd.to_numeric(df["accuracy"], errors="raise")
    if not df.empty and df["accuracy"].max() <= 1:
        df["accuracy"] *= 100

    pivot = df.pivot_table(index="k", columns="model", values="accuracy",
                           aggfunc="mean").sort_index()
    ax = pivot.plot(kind="line", marker="o", figsize=(9, 5))
    ax.set_title("Accuracy vs k-Anonymity Parameter")
    ax.set_xlabel("k value")
    ax.set_ylabel("Accuracy (%)")
    ax.grid(True, linestyle=":", alpha=0.7)
    ax.set_axisbelow(True)
    ax.legend(title="Model", frameon=False)
    fig = ax.get_figure()
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
    return ax


def plot_baseline_vs_level1(csv_path, output_path=None):
    """
    Plot the wide-format comparison file used in the current project.

    Required columns: model, baseline_accuracy, level1_accuracy.
    This is useful until results are stored in long format by anonymization level.
    """
    df = _read_results(csv_path)
    _require_columns(
        df, ["model", "baseline_accuracy", "level1_accuracy"],
        "plot_baseline_vs_level1"
    )
    df = df.copy()
    for col in ["baseline_accuracy", "level1_accuracy"]:
        df[col] = pd.to_numeric(df[col], errors="raise")
        if not df.empty and df[col].max() <= 1:
            df[col] *= 100

    plot_df = df.set_index("model")[["baseline_accuracy", "level1_accuracy"]]
    ax = plot_df.plot(kind="bar", figsize=(9, 5), edgecolor="black")
    ax.set_title("Accuracy: Baseline vs Level 1")
    ax.set_xlabel("Model")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.grid(axis="y", linestyle=":", alpha=0.7)
    ax.set_axisbelow(True)
    ax.legend(["Baseline", "Level 1"], frameon=False)
    fig = ax.get_figure()
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
    return ax
