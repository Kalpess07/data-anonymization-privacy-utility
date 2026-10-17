"""Reusable charts for privacy-utility trade-off experiments.

Expected input columns:
    model: model name, e.g. Logistic Regression or Random Forest
    anonymization_level: level label, e.g. Baseline, Level 1, Level 2, Level 3
    accuracy: numeric accuracy, either a fraction (0.855) or percentage (85.5)

Only rows present in the input data are plotted. This module does not create
or infer missing experiment results.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


LEVEL_ORDER = ["Baseline", "Level 1", "Level 2", "Level 3"]
MODEL_COLORS = {
    "Logistic Regression": "#2878B5",
    "Random Forest": "#E07A2D",
}


def _prepare_data(
    data: pd.DataFrame,
    model_col: str = "model",
    level_col: str = "anonymization_level",
    accuracy_col: str = "accuracy",
) -> pd.DataFrame:
    """Validate and normalize the input data for plotting."""
    required = {model_col, level_col, accuracy_col}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    frame = data[[model_col, level_col, accuracy_col]].copy()
    frame[accuracy_col] = pd.to_numeric(frame[accuracy_col], errors="coerce")
    frame = frame.dropna(subset=[model_col, level_col, accuracy_col])

    if frame.empty:
        raise ValueError("No valid accuracy records are available to plot.")

    # If all accuracy values are greater than 1, treat them as percentages.
    if frame[accuracy_col].gt(1).all():
        frame[accuracy_col] = frame[accuracy_col] / 100.0

    if not frame[accuracy_col].between(0, 1).all():
        raise ValueError("Accuracy values must be fractions (0-1) or percentages (0-100).")

    return frame


def plot_accuracy_by_level(
    data: pd.DataFrame,
    output_path: str | Path = "figures/accuracy_vs_anonymization_level.png",
    model_col: str = "model",
    level_col: str = "anonymization_level",
    accuracy_col: str = "accuracy",
    title: str = "Model Accuracy vs. Anonymization Level",
) -> Path:
    """Plot accuracy for each model at each level found in the input data."""
    frame = _prepare_data(data, model_col, level_col, accuracy_col)

    levels_present = frame[level_col].astype(str).unique().tolist()
    ordered_levels = [level for level in LEVEL_ORDER if level in levels_present]
    ordered_levels += sorted(level for level in levels_present if level not in ordered_levels)

    models = frame[model_col].astype(str).unique().tolist()
    x_positions = list(range(len(ordered_levels)))
    width = 0.36

    fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)

    for index, model in enumerate(models):
        model_frame = frame[frame[model_col].astype(str) == model].copy()
        model_frame[level_col] = model_frame[level_col].astype(str)
        values = []
        for level in ordered_levels:
            matches = model_frame.loc[model_frame[level_col] == level, accuracy_col]
            values.append(float(matches.iloc[0]) if not matches.empty else float("nan"))

        offset = (index - (len(models) - 1) / 2) * width
        color = MODEL_COLORS.get(model)
        bars = ax.bar(
            [x + offset for x in x_positions],
            values,
            width=width,
            label=model,
            color=color,
        )
        for bar, value in zip(bars, values):
            if pd.notna(value):
                ax.annotate(
                    f"{value:.1%}",
                    (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    ax.set_title(title, fontsize=13, pad=12)
    ax.set_xlabel("Anonymization Level")
    ax.set_ylabel("Accuracy")
    ax.set_xticks(x_positions, ordered_levels)
    ax.set_ylim(0, 1.08)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0%}"))
    ax.grid(axis="y", linestyle=":", alpha=0.45)
    ax.set_axisbelow(True)
    ax.legend(frameon=False)
    fig_path = Path(output_path)
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return fig_path


def plot_privacy_utility_template(
    data: pd.DataFrame,
    output_path: str | Path = "figures/privacy_utility_template.png",
    model_col: str = "model",
    level_col: str = "anonymization_level",
    accuracy_col: str = "accuracy",
) -> Path:
    """Create a chart template from supplied results; no values are fabricated."""
    return plot_accuracy_by_level(
        data=data,
        output_path=output_path,
        model_col=model_col,
        level_col=level_col,
        accuracy_col=accuracy_col,
        title="Privacy-Utility Trade-off: Accuracy by Anonymization Level",
    )


if __name__ == "__main__":
    # Example: read a CSV that has model, anonymization_level, and accuracy columns.
    # Adapt column names to the actual results CSV before running.
    input_file = Path("results/privacy_utility_chart_input.csv")
    if not input_file.exists():
        raise SystemExit(
            f"Input file not found: {input_file}. Prepare a CSV with columns "
            "'model', 'anonymization_level', and 'accuracy' using verified results only."
        )
    result_data = pd.read_csv(input_file)
    saved = plot_accuracy_by_level(result_data)
    print(f"Chart saved to: {saved}")
