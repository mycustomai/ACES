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


def load_combined_price_data(price_csv: Path, ar_price_csv: Path) -> tuple[list[str], list[str], dict]:
    """Load and combine price and ar_price data.

    Returns data for: Random (Low Var), Random (High Var), 1%, 5%, 10%
    Excludes absolute price reductions (10¢, 1¢).
    """
    # Load both datasets
    _, ar_tests, ar_data = load_sanity_check_data(ar_price_csv)
    _, price_tests, price_data = load_sanity_check_data(price_csv)

    # Get union of all models
    all_models = set(ar_data.keys()) | set(price_data.keys())
    model_names = sorted(all_models, key=lambda m: DISPLAY_NAME_TO_METADATA.get(m, ("", None, ""))[1] or "")

    # Combined tests: Random (Low Var), Random (High Var), 1%, 5%, 10%
    combined_tests = [
        "Random (Low Variance)",  # from ar_price index 0
        "Random (High Variance)",  # from ar_price index 1
        "1% Price Reduction",      # from price index 0
        "5% Price Reduction",      # from price index 1
        "10% Price Reduction",     # from price index 2
    ]

    combined_data = {}
    for model in model_names:
        means = []
        stderrs = []

        # Add random variance from ar_price (indices 0, 1)
        if model in ar_data:
            means.extend([ar_data[model]["means"][0], ar_data[model]["means"][1]])
            stderrs.extend([ar_data[model]["stderrs"][0], ar_data[model]["stderrs"][1]])
        else:
            means.extend([np.nan, np.nan])
            stderrs.extend([np.nan, np.nan])

        # Add percentage reductions from price (indices 0, 1, 2)
        if model in price_data:
            means.extend([price_data[model]["means"][0], price_data[model]["means"][1], price_data[model]["means"][2]])
            stderrs.extend([price_data[model]["stderrs"][0], price_data[model]["stderrs"][1], price_data[model]["stderrs"][2]])
        else:
            means.extend([np.nan, np.nan, np.nan])
            stderrs.extend([np.nan, np.nan, np.nan])

        combined_data[model] = {"means": np.array(means), "stderrs": np.array(stderrs)}

    return model_names, combined_tests, combined_data


def plot_single_task_by_provider(
    data: dict,
    task_idx: int,
    output_path: Path,
    title: str,
    task_label: str,
) -> None:
    """Create line graph for a single task with by-provider subplots.

    Each provider subplot shows one line (in that provider's color) for the task.
    """
    known = [n for n in data.keys() if n in DISPLAY_NAME_TO_METADATA]

    providers = ["Anthropic", "Google", "OpenAI"]
    provider_models = {
        p: sort_display_names_by_date(
            [n for n in known if get_provider_for_display_name(n) == p]
        )
        for p in providers
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    for ax, provider in zip(axes, providers):
        models = provider_models[provider]
        provider_color = PROVIDER_COLORS[provider]

        x_positions = []
        y_values = []
        y_errors = []

        for model_idx, model_name in enumerate(models):
            mean = data[model_name]["means"][task_idx]
            stderr = data[model_name]["stderrs"][task_idx]

            if not np.isnan(mean):
                x_positions.append(model_idx)
                y_values.append(mean)
                y_errors.append(stderr)

        if x_positions:
            # Plot single line using provider color
            ax.plot(x_positions, y_values,
                   marker='o', markersize=7, linewidth=2.5,
                   color=provider_color, label=task_label, alpha=0.9)

            # Add error bars
            ax.errorbar(x_positions, y_values, yerr=y_errors,
                       fmt='none', ecolor=provider_color, alpha=0.3,
                       capsize=4, capthick=1.5)

        # Set x-axis labels (model names with dates)
        labels = []
        for model_name in models:
            model_id, release_date, _ = DISPLAY_NAME_TO_METADATA[model_name]
            labels.append(f"{model_name}\n{release_date.strftime('%b %d, %Y')}")

        ax.set_xticks(np.arange(len(models)))
        ax.set_xticklabels(labels, rotation=0, ha='center', fontsize=9)
        ax.tick_params(axis="x", pad=10)

        # Styling
        apply_clean_style(ax)
        ax.set_title(
            provider, fontsize=13, fontweight="bold",
            color=PROVIDER_COLORS_DARK[provider], pad=12,
        )

    axes[0].set_ylabel("Failure Rate", fontsize=11, color="#555555", fontweight="medium")
    axes[1].set_ylabel("")
    axes[2].set_ylabel("")

    fig.suptitle(title, fontsize=15, fontweight="bold", color="#222222", y=1.02)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_category_combined(
    data: dict,
    task_labels: list[str],
    output_path: Path,
    title: str,
) -> None:
    """Create combined plot for all tasks within a category by provider.

    Each provider subplot shows multiple lines (one per task), all in that provider's color,
    with different line styles/markers to distinguish tasks.
    """
    known = [n for n in data.keys() if n in DISPLAY_NAME_TO_METADATA]

    providers = ["Anthropic", "Google", "OpenAI"]
    provider_models = {
        p: sort_display_names_by_date(
            [n for n in known if get_provider_for_display_name(n) == p]
        )
        for p in providers
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    # Line styles and markers for different tasks
    line_styles = ['-', '--', '-.', ':', '-']
    markers = ['o', 's', '^', 'D', 'v']

    for ax, provider in zip(axes, providers):
        models = provider_models[provider]
        provider_color = PROVIDER_COLORS[provider]

        # Plot one line per task
        for task_idx, label in enumerate(task_labels):
            x_positions = []
            y_values = []
            y_errors = []

            for model_idx, model_name in enumerate(models):
                mean = data[model_name]["means"][task_idx]
                stderr = data[model_name]["stderrs"][task_idx]

                if not np.isnan(mean):
                    x_positions.append(model_idx)
                    y_values.append(mean)
                    y_errors.append(stderr)

            if x_positions:
                linestyle = line_styles[task_idx % len(line_styles)]
                marker = markers[task_idx % len(markers)]

                # Plot line using provider color
                ax.plot(x_positions, y_values,
                       marker=marker, markersize=7, linewidth=2.5,
                       linestyle=linestyle,
                       color=provider_color, label=label, alpha=0.9)

                # Add error bars
                ax.errorbar(x_positions, y_values, yerr=y_errors,
                           fmt='none', ecolor=provider_color, alpha=0.3,
                           capsize=4, capthick=1.5)

        # Set x-axis labels (model names with dates)
        labels = []
        for model_name in models:
            model_id, release_date, _ = DISPLAY_NAME_TO_METADATA[model_name]
            labels.append(f"{model_name}\n{release_date.strftime('%b %d, %Y')}")

        ax.set_xticks(np.arange(len(models)))
        ax.set_xticklabels(labels, rotation=0, ha='center', fontsize=9)
        ax.tick_params(axis="x", pad=10)

        # Styling
        apply_clean_style(ax)
        ax.set_title(
            provider, fontsize=13, fontweight="bold",
            color=PROVIDER_COLORS_DARK[provider], pad=12,
        )

        # Add legend
        if provider == "Anthropic":  # Only add legend to first subplot
            ax.legend(fontsize=9, loc="best", framealpha=0.95, title="Test Condition")

    axes[0].set_ylabel("Failure Rate", fontsize=11, color="#555555", fontweight="medium")
    axes[1].set_ylabel("")
    axes[2].set_ylabel("")

    fig.suptitle(title, fontsize=15, fontweight="bold", color="#222222", y=1.02)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    base_output_dir = Path("artifacts/visualization/sanity_checks")

    # File paths
    rating_csv = Path("artifacts/analysis/20260121184134_rating_sanity_check.csv")
    price_csv = Path("artifacts/analysis/20260212101910_price_sanity_check.csv")
    ar_price_csv = Path("artifacts/analysis/20260213120645_ar_price_sanity_check.csv")
    instruction_csv = Path("artifacts/analysis/20260212101915_instruction_sanity_check.csv")

    # Load data
    _, rating_tests, rating_data = load_sanity_check_data(rating_csv)
    _, price_tests, price_data = load_combined_price_data(price_csv, ar_price_csv)
    _, instruction_tests, instruction_data = load_sanity_check_data(instruction_csv)

    # Individual task plots
    tasks = [
        # Rating tasks
        (rating_data, 0, "rating_plus_0.1", "Rating +0.1 sanity check failure rates by provider", "+0.1 Rating"),
        (rating_data, 1, "rating_random_low_variance", "Rating random (low variance) failure rates by provider", "Low Variance"),
        (rating_data, 2, "rating_random_high_variance", "Rating random (high variance) failure rates by provider", "High Variance"),

        # Price tasks
        (price_data, 0, "price_random_low_variance", "Price random (low variance) failure rates by provider", "Random (Low Var)"),
        (price_data, 1, "price_random_high_variance", "Price random (high variance) failure rates by provider", "Random (High Var)"),
        (price_data, 2, "price_1_percent", "Price 1% reduction failure rates by provider", "1% Reduction"),
        (price_data, 3, "price_5_percent", "Price 5% reduction failure rates by provider", "5% Reduction"),
        (price_data, 4, "price_10_percent", "Price 10% reduction failure rates by provider", "10% Reduction"),

        # Instruction tasks
        (instruction_data, 0, "instruction_budget", "Instruction following (budget) failure rates by provider", "Budget"),
        (instruction_data, 1, "instruction_color", "Instruction following (color) failure rates by provider", "Color"),
        (instruction_data, 2, "instruction_brand", "Instruction following (brand) failure rates by provider", "Brand"),
    ]

    for task_data, task_idx, filename, title, task_label in tasks:
        plot_single_task_by_provider(
            task_data,
            task_idx,
            base_output_dir / f"{filename}.png",
            title=title,
            task_label=task_label,
        )

    # Category combined plots
    plot_category_combined(
        rating_data,
        ["+0.1 Rating", "Low Variance", "High Variance"],
        base_output_dir / "rating_combined.png",
        title="Rating sanity checks combined (by release date)",
    )

    plot_category_combined(
        price_data,
        ["Random (Low Var)", "Random (High Var)", "1% Reduction", "5% Reduction", "10% Reduction"],
        base_output_dir / "price_combined.png",
        title="Price sanity checks combined (by release date)",
    )

    plot_category_combined(
        instruction_data,
        ["Budget", "Color", "Brand"],
        base_output_dir / "instruction_combined.png",
        title="Instruction following combined (by release date)",
    )
