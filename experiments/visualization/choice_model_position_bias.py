from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from experiments.visualization.common import (
    MODEL_METADATA,
    PROVIDER_COLORS,
    PROVIDER_COLORS_LIGHT,
    PROVIDER_COLORS_DARK,
    apply_clean_style,
    draw_rounded_bar,
    add_value_label,
    create_modern_legend,
)

POSITION_LABELS = ["1", "2", "3", "4", "5", "6", "7", "8"]


def load_choice_model_data(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path, index_col=0)


def compute_position_probabilities(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Compute softmax selection probabilities for each of the 8 grid positions per model."""
    results = {}
    for model_id in df.columns:
        row1 = df.loc["row_1_dummy", model_id]
        col1 = df.loc["col_1_dummy", model_id]
        col2 = df.loc["col_2_dummy", model_id]
        col3 = df.loc["col_3_dummy", model_id]

        coeffs = np.array([
            row1 + col1,  # Position 1: R1C1
            row1 + col2,  # Position 2: R1C2
            row1 + col3,  # Position 3: R1C3
            row1,         # Position 4: R1C4 (col4 = 0)
            col1,         # Position 5: R2C1 (row2 = 0)
            col2,         # Position 6: R2C2
            col3,         # Position 7: R2C3
            0.0,          # Position 8: R2C4 (both baselines)
        ])

        exp_coeffs = np.exp(coeffs - np.max(coeffs))
        probs = exp_coeffs / exp_coeffs.sum()
        results[model_id] = probs

    return results


def _sort_models_by_date(model_ids: list[str]) -> list[str]:
    return sorted(model_ids, key=lambda m: MODEL_METADATA[m][1])


def _draw_grouped_bars(
    ax: plt.Axes,
    model_ids: list[str],
    probs: dict[str, np.ndarray],
    show_position_labels: bool = True,
    show_values: bool = True,
) -> float:
    n_models = len(model_ids)
    n_positions = 8
    bar_width = 0.08
    group_width = n_positions * bar_width
    group_gap = 0.25
    x = np.arange(n_models) * (group_width + group_gap)

    rounding = 0.012
    all_heights = []
    for model_idx, model_id in enumerate(model_ids):
        provider = MODEL_METADATA[model_id][2]
        color = PROVIDER_COLORS[provider]
        light_color = PROVIDER_COLORS_LIGHT[provider]
        max_pos = int(np.argmax(probs[model_id]))

        for pos_idx in range(n_positions):
            bx = x[model_idx] + pos_idx * bar_width
            height = probs[model_id][pos_idx]
            all_heights.append(height)

            # Only draw bar if height is non-zero (skip tiny/zero bars)
            if height > 0.001:  # Threshold for ~0.1%
                # Highlight the maximum position
                is_max = (pos_idx == max_pos)
                fill = color + "30" if is_max else "white"

                draw_rounded_bar(
                    ax, bx, height, bar_width, color,
                    fill=fill,
                    rounding=rounding,
                )

            # Add value labels for all bars (lighter color for less density)
            if show_values:
                add_value_label(
                    ax, bx, height, height,
                    color="#AAAAAA",  # Lighter gray for all labels
                    fontsize=6,
                    offset_y=0.003,
                    rotation=90,
                )

    x_max = x[-1] + (n_positions - 1) * bar_width + bar_width
    ax.set_xlim(x[0] - bar_width, x_max)
    max_h = max(all_heights) if all_heights else 0.125

    if show_position_labels:
        for model_idx in range(n_models):
            for pos_idx in range(n_positions):
                bx = x[model_idx] + pos_idx * bar_width
                ax.text(
                    bx, -0.015, POSITION_LABELS[pos_idx],
                    ha="center", va="top", fontsize=6.5, color="#999999",
                    fontweight="medium",
                )

    centers = x + (n_positions - 1) * bar_width / 2
    display_names = [MODEL_METADATA[m][0] for m in model_ids]
    ax.set_xticks(centers)
    ax.set_xticklabels(display_names, rotation=0, ha="center", fontsize=9)
    ax.tick_params(axis="x", pad=20)

    # Reference line for uniform distribution
    ax.axhline(y=1 / 8, color="#999999", linestyle="--", linewidth=1.5, alpha=0.6, zorder=0)
    ax.text(
        x[0] - bar_width * 2, 1 / 8, "Uniform\n(12.5%)",
        ha="right", va="center", fontsize=8, color="#999999",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#E0E0E0", alpha=0.9),
    )

    apply_clean_style(ax)
    ax.set_ylabel("Selection probability", fontsize=11, color="#555555", fontweight="medium")

    return max_h


def plot_all_providers(csv_path: Path, output_path: Path) -> None:
    """All models on one chart, x-labels colored by provider."""
    df = load_choice_model_data(csv_path)
    probs = compute_position_probabilities(df)
    model_ids = _sort_models_by_date([m for m in df.columns if m in MODEL_METADATA])

    fig, ax = plt.subplots(figsize=(16, 5))
    max_h = _draw_grouped_bars(ax, model_ids, probs)
    ax.set_ylim(0, max_h * 1.15)

    for tick_label, model_id in zip(ax.get_xticklabels(), model_ids):
        provider = MODEL_METADATA[model_id][2]
        tick_label.set_color(PROVIDER_COLORS_DARK[provider])
        tick_label.set_fontweight("semibold")

    ax.set_title(
        "Position bias across models (by release date)",
        fontsize=14, fontweight="bold", color="#222222", pad=16,
    )

    create_modern_legend(ax, PROVIDER_COLORS, loc="upper right")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_by_provider(csv_path: Path, output_path: Path) -> None:
    """One subplot per provider."""
    df = load_choice_model_data(csv_path)
    probs = compute_position_probabilities(df)

    providers = ["Anthropic", "Google", "OpenAI"]
    provider_models = {
        p: _sort_models_by_date(
            [m for m in df.columns if m in MODEL_METADATA and MODEL_METADATA[m][2] == p]
        )
        for p in providers
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    # Collect max heights from all providers to set consistent y-axis
    max_heights = []
    for ax, provider in zip(axes, providers):
        model_ids = provider_models[provider]
        max_h = _draw_grouped_bars(ax, model_ids, probs, show_values=True)
        max_heights.append(max_h)
        ax.set_title(
            provider, fontsize=13, fontweight="bold",
            color=PROVIDER_COLORS_DARK[provider], pad=12,
        )

    # Set consistent y-axis limit across all subplots based on global max
    global_max = max(max_heights) if max_heights else 0.125
    axes[0].set_ylim(0, global_max * 1.15)  # sharey=True propagates to all axes

    axes[1].set_ylabel("")
    axes[2].set_ylabel("")

    fig.suptitle(
        "Position bias by provider (by release date)",
        fontsize=15, fontweight="bold", color="#222222", y=1.02,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    csv_path = Path("artifacts/analysis/20260122104301_choice_model.csv")
    output_dir = Path("artifacts/visualization/position_bias")

    plot_all_providers(csv_path, output_dir / "all_providers.png")
    plot_by_provider(csv_path, output_dir / "by_provider.png")
