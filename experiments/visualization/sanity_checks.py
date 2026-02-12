import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from experiments.visualization.common import (
    DISPLAY_NAME_TO_METADATA,
    PROVIDER_COLORS,
    PROVIDER_COLORS_LIGHT,
    PROVIDER_COLORS_DARK,
    apply_clean_style,
    draw_rounded_bar,
    add_value_label,
    create_modern_legend,
    sort_display_names_by_date,
    get_provider_for_display_name,
)


def _parse_value(cell: str) -> tuple[float, float]:
    """Parse 'mean (stderr)' string. Returns (mean, stderr)."""
    if cell.strip() == "---":
        return np.nan, np.nan
    match = re.match(r"([\d.]+)\s*\(([\d.]+)\)", cell.strip())
    if not match:
        return np.nan, np.nan
    return float(match.group(1)), float(match.group(2))


def load_sanity_check_data(csv_path: Path) -> tuple[list[str], list[str], dict]:
    """Load sanity check CSV. Returns (model_names, sub_test_names, data_dict).

    data_dict[model_name] = {"means": np.array, "stderrs": np.array}
    """
    df = pd.read_csv(csv_path)
    model_names = df["Model"].tolist()
    sub_tests = [c for c in df.columns if c != "Model"]

    data = {}
    for _, row in df.iterrows():
        model = row["Model"]
        means = []
        stderrs = []
        for test in sub_tests:
            m, s = _parse_value(str(row[test]))
            means.append(m)
            stderrs.append(s)
        data[model] = {"means": np.array(means), "stderrs": np.array(stderrs)}

    return model_names, sub_tests, data


def _draw_sanity_bars(
    ax: plt.Axes,
    model_names: list[str],
    sub_tests: list[str],
    data: dict,
    show_values: bool = True,
) -> float:
    n_models = len(model_names)
    n_tests = len(sub_tests)
    bar_width = 0.12
    group_width = n_tests * bar_width
    group_gap = 0.25
    x = np.arange(n_models) * (group_width + group_gap)

    all_heights = []
    label_offset_y = 0.008
    for model_idx, model_name in enumerate(model_names):
        provider = get_provider_for_display_name(model_name)
        color = PROVIDER_COLORS[provider]
        light_color = PROVIDER_COLORS_LIGHT[provider]
        means = data[model_name]["means"]
        stderrs = data[model_name]["stderrs"]
        valid_means = means[~np.isnan(means)]
        max_idx = int(np.argmax(valid_means)) if len(valid_means) > 0 else -1

        for test_idx in range(n_tests):
            bx = x[model_idx] + test_idx * bar_width
            height = means[test_idx]
            stderr = stderrs[test_idx]
            if np.isnan(height):
                continue
            # Include label offset in height calculation to prevent cutoff
            all_heights.append(height + stderr + label_offset_y)

            # Only draw bar if height is non-zero (skip tiny/zero bars)
            if height > 0.001:  # Threshold for ~0.1%
                # Draw straight bar for sanity checks
                from matplotlib.patches import Rectangle
                x_left = bx - bar_width / 2
                rect = Rectangle(
                    (x_left, 0),
                    bar_width, height,
                    facecolor="white",
                    edgecolor=color,
                    linewidth=1.0,
                    hatch="///",
                )
                ax.add_patch(rect)

            # Enhanced error bars
            if height > 0.001 and stderr > 0 and not np.isnan(stderr):
                # Main error bar line
                ax.plot(
                    [bx, bx],
                    [max(0, height - stderr), height + stderr],
                    color=color, linewidth=1.2, solid_capstyle="round",
                    alpha=0.8, zorder=5,
                )
                # Caps
                cap_width = bar_width * 0.25
                ax.plot(
                    [bx - cap_width, bx + cap_width],
                    [height + stderr, height + stderr],
                    color=color, linewidth=1.2, solid_capstyle="round",
                    alpha=0.8, zorder=5,
                )
                ax.plot(
                    [bx - cap_width, bx + cap_width],
                    [max(0, height - stderr), max(0, height - stderr)],
                    color=color, linewidth=1.2, solid_capstyle="round",
                    alpha=0.8, zorder=5,
                )

            # Add value labels for all bars including 0%
            if show_values:
                add_value_label(
                    ax, bx, height + stderr, height,
                    color="#666666",
                    fontsize=7,
                    offset_y=label_offset_y,
                    skip_zero=False,
                )

    # sub-test labels below bars
    for model_idx in range(n_models):
        for test_idx in range(n_tests):
            bx = x[model_idx] + test_idx * bar_width
            ax.text(
                bx, -0.01, str(test_idx + 1),
                ha="center", va="top", fontsize=7, color="#999999",
                fontweight="medium",
            )

    centers = x + (n_tests - 1) * bar_width / 2
    ax.set_xticks(centers)
    ax.set_xticklabels(model_names, rotation=0, ha="center", fontsize=9)
    ax.tick_params(axis="x", pad=16)

    max_h = max(all_heights) if all_heights else 0.1
    x_max = x[-1] + (n_tests - 1) * bar_width + bar_width
    ax.set_xlim(x[0] - bar_width, x_max)

    apply_clean_style(ax)
    ax.set_ylabel("Failure rate", fontsize=11, color="#555555", fontweight="medium")

    return max_h


def plot_all_providers(
    csv_path: Path,
    output_path: Path,
    title: str,
    sub_test_labels: list[str] | None = None,
) -> None:
    model_names, sub_tests, data = load_sanity_check_data(csv_path)
    # filter to known models and sort by date
    known = [n for n in model_names if n in DISPLAY_NAME_TO_METADATA]
    known = sort_display_names_by_date(known)

    fig, ax = plt.subplots(figsize=(16, 5))
    max_h = _draw_sanity_bars(ax, known, sub_tests, data)
    # Add extra space for text height above labels (1.15x for more room)
    ax.set_ylim(0, max_h * 1.15)

    for tick_label, model_name in zip(ax.get_xticklabels(), known):
        provider = get_provider_for_display_name(model_name)
        tick_label.set_color(PROVIDER_COLORS_DARK[provider])
        tick_label.set_fontweight("semibold")

    ax.set_title(title, fontsize=14, fontweight="bold", color="#222222", pad=16)

    # provider legend
    create_modern_legend(ax, PROVIDER_COLORS, loc="upper right")

    # sub-test legend annotation
    labels = sub_test_labels or sub_tests
    label_text = "  ".join(f"{i+1}: {l}" for i, l in enumerate(labels))
    fig.text(
        0.5, -0.02, label_text,
        ha="center", fontsize=8, color="#666666",
    )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_by_provider(
    csv_path: Path,
    output_path: Path,
    title: str,
    sub_test_labels: list[str] | None = None,
) -> None:
    model_names, sub_tests, data = load_sanity_check_data(csv_path)
    known = [n for n in model_names if n in DISPLAY_NAME_TO_METADATA]

    providers = ["Anthropic", "Google", "OpenAI"]
    provider_models = {
        p: sort_display_names_by_date(
            [n for n in known if get_provider_for_display_name(n) == p]
        )
        for p in providers
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    # Collect max heights from all providers to set consistent y-axis
    max_heights = []
    for ax, provider in zip(axes, providers):
        models = provider_models[provider]
        max_h = _draw_sanity_bars(ax, models, sub_tests, data, show_values=True)
        max_heights.append(max_h)
        ax.set_title(
            provider, fontsize=13, fontweight="bold",
            color=PROVIDER_COLORS_DARK[provider], pad=12,
        )

    # Set consistent y-axis limit across all subplots based on global max
    global_max = max(max_heights) if max_heights else 0.1
    axes[0].set_ylim(0, global_max * 1.15)  # sharey=True propagates to all axes

    axes[1].set_ylabel("")
    axes[2].set_ylabel("")

    fig.suptitle(title, fontsize=15, fontweight="bold", color="#222222", y=1.02)

    labels = sub_test_labels or sub_tests
    label_text = "  ".join(f"{i+1}: {l}" for i, l in enumerate(labels))
    fig.text(
        0.5, -0.02, label_text,
        ha="center", fontsize=8, color="#666666",
    )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output_path}")


SANITY_CHECK_CONFIGS = {
    "rating": {
        "csv": "artifacts/analysis/20260121184134_rating_sanity_check.csv",
        "title_all": "Rating sanity check failure rates (by release date)",
        "title_provider": "Rating sanity check failure rates by provider (by release date)",
        "labels": ["+0.1 Rating", "Low Variance", "High Variance"],
        "prefix": "rating_sanity",
    },
    "price": {
        "csv": "artifacts/analysis/20260212101910_price_sanity_check.csv",
        "title_all": "Price sanity check failure rates (by release date)",
        "title_provider": "Price sanity check failure rates by provider (by release date)",
        "labels": ["1% Reduction", "5% Reduction", "10% Reduction"],
        "prefix": "price_sanity",
    },
    "instruction": {
        "csv": "artifacts/analysis/20260212101915_instruction_sanity_check.csv",
        "title_all": "Instruction following failure rates (by release date)",
        "title_provider": "Instruction following failure rates by provider (by release date)",
        "labels": ["Budget", "Color", "Brand"],
        "prefix": "instruction_sanity",
    },
}


if __name__ == "__main__":
    output_dir = Path("artifacts/visualization")

    for key, cfg in SANITY_CHECK_CONFIGS.items():
        csv_path = Path(cfg["csv"])
        plot_all_providers(
            csv_path,
            output_dir / f"{cfg['prefix']}_all_providers.png",
            title=cfg["title_all"],
            sub_test_labels=cfg["labels"],
        )
        plot_by_provider(
            csv_path,
            output_dir / f"{cfg['prefix']}_by_provider.png",
            title=cfg["title_provider"],
            sub_test_labels=cfg["labels"],
        )
