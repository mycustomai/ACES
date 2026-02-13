from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.visualization.common import (
    MODEL_METADATA,
    PROVIDER_COLORS,
    PROVIDER_COLORS_DARK,
    PROVIDER_COLORS_LIGHT,
    apply_clean_style,
    create_modern_legend,
    draw_rounded_bar,
)


def load_choice_model_coefficients(csv_path: Path) -> pd.DataFrame:
    """Load choice model coefficients."""
    return pd.read_csv(csv_path, index_col=0)


def compute_probability_change(
    baseline_prob: float,
    coefficient: float,
    n_alternatives: int = 8,
) -> float:
    """
    Compute new selection probability after adding a feature.

    Args:
        baseline_prob: Baseline selection probability (e.g., 0.1 for 10%)
        coefficient: Choice model coefficient for the feature
        n_alternatives: Number of alternatives in choice set (default 8)

    Returns:
        New selection probability after adding the feature
    """
    # Assuming other products have equal probability
    other_prob = (1 - baseline_prob) / (n_alternatives - 1)

    # Compute baseline utility difference (product vs average other)
    # baseline_prob = exp(U0) / (exp(U0) + (n-1)*exp(U_other))
    # Rearranging: exp(U0 - U_other) = baseline_prob / other_prob
    baseline_utility_diff = np.log(baseline_prob / other_prob)

    # New utility difference after adding feature
    new_utility_diff = baseline_utility_diff + coefficient

    # Convert back to probability
    # p_new = exp(U_new) / (exp(U_new) + (n-1)*exp(U_other))
    # p_new = exp(U_new - U_other) / (exp(U_new - U_other) + (n-1))
    new_prob = np.exp(new_utility_diff) / (np.exp(new_utility_diff) + (n_alternatives - 1))

    return new_prob


def compute_price_impact(
    baseline_prob: float,
    price_coefficient: float,
    price_percentage_change: float,
    n_alternatives: int = 8,
) -> float:
    """
    Compute new selection probability after changing price by a percentage.

    Args:
        baseline_prob: Baseline selection probability
        price_coefficient: Choice model coefficient for log_price
        price_percentage_change: Percentage change in price (e.g., -0.05 for -5%)
        n_alternatives: Number of alternatives in choice set

    Returns:
        New selection probability after price change
    """
    # Utility change from percentage price change
    # new_price = old_price * (1 + price_percentage_change)
    # utility_change = coefficient * log(new_price / old_price)
    #                = coefficient * log(1 + price_percentage_change)
    utility_change = price_coefficient * np.log(1 + price_percentage_change)

    other_prob = (1 - baseline_prob) / (n_alternatives - 1)
    baseline_utility_diff = np.log(baseline_prob / other_prob)
    new_utility_diff = baseline_utility_diff + utility_change
    new_prob = np.exp(new_utility_diff) / (np.exp(new_utility_diff) + (n_alternatives - 1))

    return new_prob


def compute_rating_impact(
    baseline_prob: float,
    rating_coefficient: float,
    rating_change: float,
    n_alternatives: int = 8,
) -> float:
    """
    Compute new selection probability after changing rating.

    Args:
        baseline_prob: Baseline selection probability
        rating_coefficient: Choice model coefficient for rating
        rating_change: Change in rating (e.g., 0.1)
        n_alternatives: Number of alternatives in choice set

    Returns:
        New selection probability after rating change
    """
    utility_change = rating_coefficient * rating_change

    other_prob = (1 - baseline_prob) / (n_alternatives - 1)
    baseline_utility_diff = np.log(baseline_prob / other_prob)
    new_utility_diff = baseline_utility_diff + utility_change
    new_prob = np.exp(new_utility_diff) / (np.exp(new_utility_diff) + (n_alternatives - 1))

    return new_prob


def _sort_models_by_date(model_ids: list[str]) -> list[str]:
    """Sort model IDs by release date."""
    return sorted(model_ids, key=lambda m: MODEL_METADATA[m][1])


def plot_feature_impact_all_providers(
    csv_path: Path,
    output_path: Path,
    feature_name: str,
    coefficient_name: str,
    rating_change: float | None = None,
    price_percentage_change: float | None = None,
    baseline_prob: float = 0.1,
) -> None:
    """
    Plot how selection probability changes for a single feature across all models.

    Shows baseline probability and change for all models, sorted by release date.
    X-axis labels are colored by provider.
    """
    df = load_choice_model_coefficients(csv_path)

    # Get all models that are in MODEL_METADATA
    model_ids = [m for m in df.columns if m in MODEL_METADATA]
    model_ids = _sort_models_by_date(model_ids)

    n_models = len(model_ids)

    fig, ax = plt.subplots(figsize=(16, 5))

    bar_width = 0.65
    x = np.arange(n_models)

    # Compute probabilities for all models
    probabilities = []
    changes = []

    for model_id in model_ids:
        if rating_change is not None:
            # Rating change
            new_prob = compute_rating_impact(
                baseline_prob,
                df.loc[coefficient_name, model_id],
                rating_change,
            )
        elif price_percentage_change is not None:
            # Price percentage change
            new_prob = compute_price_impact(
                baseline_prob,
                df.loc[coefficient_name, model_id],
                price_percentage_change,
            )
        else:
            # Tag addition (binary feature)
            new_prob = compute_probability_change(
                baseline_prob,
                df.loc[coefficient_name, model_id],
            )

        probabilities.append(new_prob)
        changes.append(new_prob - baseline_prob)

    # Plot bars for each model
    for model_idx, (model_id, prob, change) in enumerate(zip(model_ids, probabilities, changes)):
        provider = MODEL_METADATA[model_id][2]
        color = PROVIDER_COLORS[provider]
        bx = x[model_idx]

        # Draw bar to actual probability height (can be above or below baseline)
        # If change is positive, use light fill; otherwise white
        if change > 0:
            fill = PROVIDER_COLORS_LIGHT[provider] + "40"  # Light transparent fill
        else:
            fill = "white"

        draw_rounded_bar(
            ax, bx, prob, bar_width, color,
            fill=fill,
            rounding=0.012,
        )

        # Draw baseline indicator line (dashed horizontal line at baseline)
        baseline_x_left = bx - bar_width / 2
        baseline_x_right = bx + bar_width / 2
        ax.plot(
            [baseline_x_left, baseline_x_right],
            [baseline_prob, baseline_prob],
            color="#666666", linewidth=1.5, linestyle="--",
            alpha=0.7, zorder=10,
        )

        # Add value label
        label_y = prob + 0.005 if prob > baseline_prob else prob - 0.005
        label_va = "bottom" if prob > baseline_prob else "top"
        ax.text(
            bx, label_y,
            f"{prob:.1%}",
            ha="center", va=label_va,
            fontsize=8, color="#333333",
            fontweight="medium",
        )

        # Add change label (smaller, inside the bar or between baseline and bar)
        if abs(change) > 0.02:  # Only show if change is significant enough
            change_label = f"{change*100:+.1f} p.p."
            # Position the label in the middle of the change (between baseline and new prob)
            if change > 0:
                change_y = baseline_prob + change / 2
            else:
                change_y = prob + abs(change) / 2
            ax.text(
                bx, change_y,
                change_label,
                ha="center", va="center",
                fontsize=7, color="#333333",
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#DDDDDD", alpha=0.95),
            )

    # Baseline reference line
    ax.axhline(y=baseline_prob, color="#666666", linestyle="--", linewidth=1.5, alpha=0.5, zorder=0)
    ax.text(
        -0.5, baseline_prob, f"Baseline\n({baseline_prob:.0%})",
        ha="right", va="center", fontsize=8, color="#666666",
    )

    # Set x-ticks and labels (model names with dates, colored by provider)
    labels = [f"{MODEL_METADATA[m][0]}\n{MODEL_METADATA[m][1].strftime('%b %d, %Y')}" for m in model_ids]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, ha="center", fontsize=9)
    ax.tick_params(axis="x", pad=10)

    # Color x-tick labels by provider
    for tick_label, model_id in zip(ax.get_xticklabels(), model_ids):
        provider = MODEL_METADATA[model_id][2]
        tick_label.set_color(PROVIDER_COLORS_DARK[provider])
        tick_label.set_fontweight("semibold")

    # Set axis limits (account for labels above/below bars)
    max_prob = max(probabilities + [baseline_prob]) + 0.02
    min_prob = 0  # Keep y-axis starting at 0
    ax.set_xlim(x[0] - bar_width, x[-1] + bar_width)
    ax.set_ylim(min_prob, max_prob * 1.1)

    apply_clean_style(ax)
    ax.set_ylabel("Selection probability", fontsize=11, color="#555555", fontweight="medium")

    # Create title with feature name
    title = f"Impact of {feature_name} on selection probability (by release date)"
    ax.set_title(title, fontsize=14, fontweight="bold", color="#222222", pad=16)

    # Provider legend
    create_modern_legend(ax, PROVIDER_COLORS, loc="upper right")

    # Add note about assumptions
    note_text = f"Baseline: {baseline_prob:.0%} selection | Gray = baseline, Color = change from {feature_name}"
    fig.text(
        0.5, -0.02, note_text,
        ha="center", fontsize=8, color="#666666",
    )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_feature_impact_by_provider(
    csv_path: Path,
    output_path: Path,
    feature_name: str,
    coefficient_name: str,
    rating_change: float | None = None,
    price_percentage_change: float | None = None,
    baseline_prob: float = 0.1,
) -> None:
    """
    Plot how selection probability changes for a single feature by provider.

    Creates 3 subplots (one per provider) showing models sorted by release date.
    """
    df = load_choice_model_coefficients(csv_path)

    # Get all models and group by provider
    all_model_ids = [m for m in df.columns if m in MODEL_METADATA]
    providers = ["Anthropic", "Google", "OpenAI"]
    provider_models = {
        p: _sort_models_by_date(
            [m for m in all_model_ids if MODEL_METADATA[m][2] == p]
        )
        for p in providers
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    max_probs_all = []

    for ax, provider in zip(axes, providers):
        model_ids = provider_models[provider]
        if not model_ids:
            continue

        n_models = len(model_ids)
        bar_width = 0.65
        x = np.arange(n_models)

        # Compute probabilities
        probabilities = []
        changes = []

        for model_id in model_ids:
            if rating_change is not None:
                new_prob = compute_rating_impact(
                    baseline_prob,
                    df.loc[coefficient_name, model_id],
                    rating_change,
                )
            elif price_percentage_change is not None:
                new_prob = compute_price_impact(
                    baseline_prob,
                    df.loc[coefficient_name, model_id],
                    price_percentage_change,
                )
            else:
                new_prob = compute_probability_change(
                    baseline_prob,
                    df.loc[coefficient_name, model_id],
                )

            probabilities.append(new_prob)
            changes.append(new_prob - baseline_prob)

        max_probs_all.extend(probabilities)

        # Plot bars
        color = PROVIDER_COLORS[provider]
        for model_idx, (model_id, prob, change) in enumerate(zip(model_ids, probabilities, changes)):
            bx = x[model_idx]

            # Draw bar to actual probability height (can be above or below baseline)
            if change > 0:
                fill = PROVIDER_COLORS_LIGHT[provider] + "40"
            else:
                fill = "white"

            draw_rounded_bar(
                ax, bx, prob, bar_width, color,
                fill=fill,
                rounding=0.012,
            )

            # Draw baseline indicator line
            baseline_x_left = bx - bar_width / 2
            baseline_x_right = bx + bar_width / 2
            ax.plot(
                [baseline_x_left, baseline_x_right],
                [baseline_prob, baseline_prob],
                color="#666666", linewidth=1.5, linestyle="--",
                alpha=0.7, zorder=10,
            )

            # Add value label
            label_y = prob + 0.005 if prob > baseline_prob else prob - 0.005
            label_va = "bottom" if prob > baseline_prob else "top"
            ax.text(
                bx, label_y,
                f"{prob:.1%}",
                ha="center", va=label_va,
                fontsize=8, color="#333333",
                fontweight="medium",
            )

            # Add change label
            if abs(change) > 0.02:
                change_label = f"{change*100:+.1f} p.p."
                # Position the label in the middle of the change
                if change > 0:
                    change_y = baseline_prob + change / 2
                else:
                    change_y = prob + abs(change) / 2
                ax.text(
                    bx, change_y,
                    change_label,
                    ha="center", va="center",
                    fontsize=7, color="#333333",
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#DDDDDD", alpha=0.95),
                )

        # Set x-ticks and labels (model names with dates)
        labels = [f"{MODEL_METADATA[m][0]}\n{MODEL_METADATA[m][1].strftime('%b %d, %Y')}" for m in model_ids]
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0, ha="center", fontsize=9)
        ax.tick_params(axis="x", pad=10)

        # Set axis limits
        ax.set_xlim(x[0] - bar_width, x[-1] + bar_width)

        apply_clean_style(ax)
        ax.set_title(
            provider, fontsize=13, fontweight="bold",
            color=PROVIDER_COLORS_DARK[provider], pad=12,
        )

    # Set consistent y-axis for all subplots (account for labels)
    max_prob = max(max_probs_all + [baseline_prob]) + 0.02
    axes[0].set_ylim(0, max_prob * 1.1)

    # Reference line for baseline
    for ax in axes:
        ax.axhline(y=baseline_prob, color="#999999", linestyle=":", linewidth=1.0, alpha=0.4, zorder=0)

    axes[0].set_ylabel("Selection probability", fontsize=11, color="#555555", fontweight="medium")
    axes[1].set_ylabel("")
    axes[2].set_ylabel("")

    # Main title
    title = f"Impact of {feature_name} on selection probability by provider (by release date)"
    fig.suptitle(title, fontsize=15, fontweight="bold", color="#222222", y=1.02)

    # Add note
    note_text = f"Baseline: {baseline_prob:.0%} | Dashed line = baseline"
    fig.text(
        0.5, -0.02, note_text,
        ha="center", fontsize=8, color="#666666",
    )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_rating_impact_by_provider(csv_path: Path, output_dir: Path, baseline_prob: float = 0.1) -> None:
    """Generate rating impact visualizations (both all_providers and by_provider)."""
    plot_feature_impact_all_providers(
        csv_path,
        output_dir / "rating_increase_all_providers.png",
        feature_name="Rating +0.1",
        coefficient_name="rating",
        rating_change=0.1,
        baseline_prob=baseline_prob,
    )
    plot_feature_impact_by_provider(
        csv_path,
        output_dir / "rating_increase_by_provider.png",
        feature_name="Rating +0.1",
        coefficient_name="rating",
        rating_change=0.1,
        baseline_prob=baseline_prob,
    )


def plot_price_impact_by_provider(csv_path: Path, output_dir: Path, baseline_prob: float = 0.1) -> None:
    """Generate price impact visualizations (both all_providers and by_provider)."""
    plot_feature_impact_all_providers(
        csv_path,
        output_dir / "price_decrease_all_providers.png",
        feature_name="Price -5%",
        coefficient_name="log_price",
        price_percentage_change=-0.05,
        baseline_prob=baseline_prob,
    )
    plot_feature_impact_by_provider(
        csv_path,
        output_dir / "price_decrease_by_provider.png",
        feature_name="Price -5%",
        coefficient_name="log_price",
        price_percentage_change=-0.05,
        baseline_prob=baseline_prob,
    )


def plot_tags_impact_by_provider(csv_path: Path, output_dir: Path, baseline_prob: float = 0.1) -> None:
    """Generate tags impact visualizations (both all_providers and by_provider for each tag)."""
    # Sponsored tag - both versions
    plot_feature_impact_all_providers(
        csv_path,
        output_dir / "sponsored_tag_all_providers.png",
        feature_name="Sponsored Tag",
        coefficient_name="sponsored_tag",
        baseline_prob=baseline_prob,
    )
    plot_feature_impact_by_provider(
        csv_path,
        output_dir / "sponsored_tag_by_provider.png",
        feature_name="Sponsored Tag",
        coefficient_name="sponsored_tag",
        baseline_prob=baseline_prob,
    )

    # Overall pick tag - both versions
    plot_feature_impact_all_providers(
        csv_path,
        output_dir / "overall_pick_all_providers.png",
        feature_name="Overall Pick",
        coefficient_name="overall_pick_tag",
        baseline_prob=baseline_prob,
    )
    plot_feature_impact_by_provider(
        csv_path,
        output_dir / "overall_pick_by_provider.png",
        feature_name="Overall Pick",
        coefficient_name="overall_pick_tag",
        baseline_prob=baseline_prob,
    )


def generate_all_feature_impact_plots(csv_path: Path, output_dir: Path, baseline_prob: float = 0.1) -> None:
    """Generate all feature impact plots."""
    features = [
        ("Sponsored Tag", "sponsored_tag", None, None, "sponsored_tag"),
        ("Overall Pick", "overall_pick_tag", None, None, "overall_pick"),
        ("Rating +0.1", "rating", 0.1, None, "rating_increase"),
        ("Price -5%", "log_price", None, -0.05, "price_decrease"),
    ]

    for feature_name, coefficient_name, rating_change, price_percentage_change, filename in features:
        # All providers plot
        plot_feature_impact_all_providers(
            csv_path,
            output_dir / f"{filename}_all_providers.png",
            feature_name=feature_name,
            coefficient_name=coefficient_name,
            rating_change=rating_change,
            price_percentage_change=price_percentage_change,
            baseline_prob=baseline_prob,
        )

        # By provider plot
        plot_feature_impact_by_provider(
            csv_path,
            output_dir / f"{filename}_by_provider.png",
            feature_name=feature_name,
            coefficient_name=coefficient_name,
            rating_change=rating_change,
            price_percentage_change=price_percentage_change,
            baseline_prob=baseline_prob,
        )


if __name__ == "__main__":
    csv_path = Path("artifacts/analysis/20260122104301_choice_model.csv")
    output_dir = Path("artifacts/visualization/feature_impact")
    generate_all_feature_impact_plots(csv_path, output_dir)
