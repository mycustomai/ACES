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


def compute_price_equivalent(beta_z: float, delta_z: float, beta_price: float) -> float:
    """
    Compute price-equivalent trade-off.

    Formula: λ = exp(-β_z Δz / β_price)
    Returns: Percentage change in price (λ - 1) × 100%

    Args:
        beta_z: Coefficient for feature
        delta_z: Change in feature
        beta_price: Coefficient for log_price

    Returns:
        Percentage change in price
    """
    lambda_val = np.exp(-beta_z * delta_z / beta_price)
    return (lambda_val - 1) * 100


def _sort_models_by_date(model_ids: list[str]) -> list[str]:
    """Sort model IDs by release date."""
    return sorted(model_ids, key=lambda m: MODEL_METADATA[m][1])


def plot_single_feature_all_providers(
    csv_path: Path,
    output_path: Path,
    feature_name: str,
    coefficient_name: str,
    delta_z: float,
) -> None:
    """
    Plot price-equivalent trade-off for a single feature across all models.

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

    # Compute price-equivalent percentages for all models
    percentages = []
    for model_id in model_ids:
        beta_z = df.loc[coefficient_name, model_id]
        beta_price = df.loc["log_price", model_id]
        pct = compute_price_equivalent(beta_z, delta_z, beta_price)
        percentages.append(pct)

    # Plot bars for each model
    for model_idx, (model_id, pct) in enumerate(zip(model_ids, percentages)):
        provider = MODEL_METADATA[model_id][2]
        color = PROVIDER_COLORS[provider]
        light_color = PROVIDER_COLORS_LIGHT[provider]
        bx = x[model_idx]

        # Determine fill based on positive/negative
        if pct >= 0:
            fill = light_color + "40"  # Light transparent fill for positive
        else:
            fill = "white"  # White fill for negative

        # Draw bar
        draw_rounded_bar(
            ax, bx, pct, bar_width, color,
            fill=fill,
            rounding=0.012,
        )

        # Add value label
        label_y = pct + (1 if pct >= 0 else -1)
        label_va = "bottom" if pct >= 0 else "top"
        ax.text(
            bx, label_y,
            f"{pct:+.1f}%",
            ha="center", va=label_va,
            fontsize=9, color="#333333",
            fontweight="medium",
        )

    # Zero reference line
    ax.axhline(y=0, color="#666666", linestyle="-", linewidth=1.5, alpha=0.5, zorder=0)

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

    # Set axis limits
    max_pct = max(abs(min(percentages)), abs(max(percentages))) if percentages else 0
    y_limit = max_pct * 1.15
    ax.set_xlim(x[0] - bar_width, x[-1] + bar_width)
    ax.set_ylim(-y_limit, y_limit)

    apply_clean_style(ax)
    ax.set_ylabel("Price change (%)", fontsize=11, color="#555555", fontweight="medium")

    # Format y-axis ticks to show percentage values
    from matplotlib.ticker import FuncFormatter
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{int(y)}'))

    # Create title
    title = f"Price-equivalent trade-off: {feature_name} (by release date)"
    ax.set_title(title, fontsize=14, fontweight="bold", color="#222222", pad=16)

    # Provider legend
    create_modern_legend(ax, PROVIDER_COLORS, loc="upper right")

    # Add note
    note_text = "Positive = can raise price | Negative = must cut price"
    fig.text(
        0.5, -0.02, note_text,
        ha="center", fontsize=8, color="#666666",
    )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_single_feature_by_provider(
    csv_path: Path,
    output_path: Path,
    feature_name: str,
    coefficient_name: str,
    delta_z: float,
) -> None:
    """
    Plot price-equivalent trade-off for a single feature by provider.

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

    max_pcts_all = []

    for ax, provider in zip(axes, providers):
        model_ids = provider_models[provider]
        if not model_ids:
            continue

        n_models = len(model_ids)
        bar_width = 0.65
        x = np.arange(n_models)

        # Compute percentages
        percentages = []
        for model_id in model_ids:
            beta_z = df.loc[coefficient_name, model_id]
            beta_price = df.loc["log_price", model_id]
            pct = compute_price_equivalent(beta_z, delta_z, beta_price)
            percentages.append(pct)

        max_pcts_all.extend([abs(p) for p in percentages])

        # Plot bars
        color = PROVIDER_COLORS[provider]
        light_color = PROVIDER_COLORS_LIGHT[provider]

        for model_idx, (model_id, pct) in enumerate(zip(model_ids, percentages)):
            bx = x[model_idx]

            # Determine fill
            if pct >= 0:
                fill = light_color + "40"
            else:
                fill = "white"

            draw_rounded_bar(
                ax, bx, pct, bar_width, color,
                fill=fill,
                rounding=0.012,
            )

            # Add value label
            label_y = pct + (1 if pct >= 0 else -1)
            label_va = "bottom" if pct >= 0 else "top"
            ax.text(
                bx, label_y,
                f"{pct:+.1f}%",
                ha="center", va=label_va,
                fontsize=9, color="#333333",
                fontweight="medium",
            )

        # Zero reference line
        ax.axhline(y=0, color="#666666", linestyle="-", linewidth=1.5, alpha=0.5, zorder=0)

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

    # Set consistent y-axis for all subplots
    max_pct = max(max_pcts_all) if max_pcts_all else 0
    y_limit = max_pct * 1.15
    axes[0].set_ylim(-y_limit, y_limit)

    # Format y-axis ticks to show percentage values
    from matplotlib.ticker import FuncFormatter
    for ax in axes:
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{int(y)}'))

    axes[0].set_ylabel("Price change (%)", fontsize=11, color="#555555", fontweight="medium")
    axes[1].set_ylabel("")
    axes[2].set_ylabel("")

    # Main title
    title = f"Price-equivalent trade-off: {feature_name} by provider (by release date)"
    fig.suptitle(title, fontsize=15, fontweight="bold", color="#222222", y=1.02)

    # Add note
    note_text = "Positive = can raise price | Negative = must cut price"
    fig.text(
        0.5, -0.02, note_text,
        ha="center", fontsize=8, color="#666666",
    )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_combined_all_providers(
    csv_path: Path,
    output_path: Path,
) -> None:
    """
    Plot price-equivalent trade-offs for all features across all models.

    Uses different hatch patterns for different features.
    """
    df = load_choice_model_coefficients(csv_path)

    # Define features with (name, coefficient, delta_z, hatch_pattern)
    features = [
        ("Overall Pick", "overall_pick_tag", 1.0, "///"),
        ("Rating +0.1", "rating", 0.1, "\\\\\\"),
        ("Double Reviews", "log_rating_count", np.log(2), "..."),
        ("Sponsored Tag", "sponsored_tag", 1.0, "xxx"),
    ]

    # Get all models
    model_ids = [m for m in df.columns if m in MODEL_METADATA]
    model_ids = _sort_models_by_date(model_ids)

    n_models = len(model_ids)
    n_features = len(features)

    fig, ax = plt.subplots(figsize=(18, 6))

    bar_width = 0.18
    group_gap = 0.1

    # Calculate x positions for grouped bars
    x_base = np.arange(n_models) * (n_features * bar_width + group_gap)

    # Plot bars for each feature
    for feat_idx, (feat_name, coef_name, delta_z, hatch) in enumerate(features):
        x = x_base + feat_idx * bar_width
        percentages = []

        for model_id in model_ids:
            beta_z = df.loc[coef_name, model_id]
            beta_price = df.loc["log_price", model_id]
            pct = compute_price_equivalent(beta_z, delta_z, beta_price)
            percentages.append(pct)

        # Plot bars with provider colors and feature-specific hatches
        for model_idx, (model_id, pct) in enumerate(zip(model_ids, percentages)):
            provider = MODEL_METADATA[model_id][2]
            color = PROVIDER_COLORS[provider]
            bx = x[model_idx]

            # Draw bar with hatch pattern
            draw_rounded_bar(
                ax, bx, pct, bar_width, color,
                fill="white",
                rounding=0.012,
                hatch=hatch,
            )

    # Zero reference line
    ax.axhline(y=0, color="#666666", linestyle="-", linewidth=1.5, alpha=0.5, zorder=0)

    # Set x-ticks at center of each group
    x_centers = x_base + (n_features - 1) * bar_width / 2
    labels = [f"{MODEL_METADATA[m][0]}\n{MODEL_METADATA[m][1].strftime('%b %d, %Y')}" for m in model_ids]
    ax.set_xticks(x_centers)
    ax.set_xticklabels(labels, rotation=0, ha="center", fontsize=9)
    ax.tick_params(axis="x", pad=10)

    # Color x-tick labels by provider
    for tick_label, model_id in zip(ax.get_xticklabels(), model_ids):
        provider = MODEL_METADATA[model_id][2]
        tick_label.set_color(PROVIDER_COLORS_DARK[provider])
        tick_label.set_fontweight("semibold")

    # Set axis limits
    ax.set_xlim(x_base[0] - bar_width, x_base[-1] + n_features * bar_width)
    ax.set_ylim(-40, 1000)

    apply_clean_style(ax)
    ax.set_ylabel("Price change (%)", fontsize=11, color="#555555", fontweight="medium")

    # Format y-axis ticks to show percentage values
    from matplotlib.ticker import FuncFormatter
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{int(y)}'))

    # Create title
    title = "Price-equivalent trade-offs: All features (by release date)"
    ax.set_title(title, fontsize=14, fontweight="bold", color="#222222", pad=16)

    # Create dual legend: providers (colors) and features (hatches)
    # Provider legend
    from matplotlib.patches import Patch
    provider_handles = [
        Patch(facecolor=PROVIDER_COLORS[p], edgecolor="#333333", label=p)
        for p in ["Anthropic", "Google", "OpenAI"]
    ]

    # Feature legend (with hatches)
    feature_handles = [
        Patch(facecolor="white", edgecolor="#333333", hatch=hatch, label=name)
        for name, _, _, hatch in features
    ]

    # Two legends
    legend1 = ax.legend(handles=provider_handles, loc="upper left", title="Provider",
                       framealpha=0.95, fontsize=9)
    ax.add_artist(legend1)  # Add first legend back
    ax.legend(handles=feature_handles, loc="upper right", title="Feature",
             framealpha=0.95, fontsize=9)

    # Add note
    note_text = "Positive = can raise price | Negative = must cut price"
    fig.text(
        0.5, -0.02, note_text,
        ha="center", fontsize=8, color="#666666",
    )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_combined_by_provider(
    csv_path: Path,
    output_path: Path,
) -> None:
    """
    Plot price-equivalent trade-offs for all features by provider.

    Creates 3 subplots with different hatch patterns for different features.
    """
    df = load_choice_model_coefficients(csv_path)

    # Define features
    features = [
        ("Overall Pick", "overall_pick_tag", 1.0, "///"),
        ("Rating +0.1", "rating", 0.1, "\\\\\\"),
        ("Double Reviews", "log_rating_count", np.log(2), "..."),
        ("Sponsored Tag", "sponsored_tag", 1.0, "xxx"),
    ]

    # Get all models and group by provider
    all_model_ids = [m for m in df.columns if m in MODEL_METADATA]
    providers = ["Anthropic", "Google", "OpenAI"]
    provider_models = {
        p: _sort_models_by_date(
            [m for m in all_model_ids if MODEL_METADATA[m][2] == p]
        )
        for p in providers
    }

    fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=True)

    n_features = len(features)
    bar_width = 0.18
    group_gap = 0.08

    for ax, provider in zip(axes, providers):
        model_ids = provider_models[provider]
        if not model_ids:
            continue

        n_models = len(model_ids)
        color = PROVIDER_COLORS[provider]

        # Calculate x positions
        x_base = np.arange(n_models) * (n_features * bar_width + group_gap)

        # Plot bars for each feature
        for feat_idx, (feat_name, coef_name, delta_z, hatch) in enumerate(features):
            x = x_base + feat_idx * bar_width
            percentages = []

            for model_id in model_ids:
                beta_z = df.loc[coef_name, model_id]
                beta_price = df.loc["log_price", model_id]
                pct = compute_price_equivalent(beta_z, delta_z, beta_price)
                percentages.append(pct)

            # Plot bars
            for model_idx, (model_id, pct) in enumerate(zip(model_ids, percentages)):
                bx = x[model_idx]

                draw_rounded_bar(
                    ax, bx, pct, bar_width, color,
                    fill="white",
                    rounding=0.012,
                    hatch=hatch,
                )

        # Zero reference line
        ax.axhline(y=0, color="#666666", linestyle="-", linewidth=1.5, alpha=0.5, zorder=0)

        # Set x-ticks at center of each group
        x_centers = x_base + (n_features - 1) * bar_width / 2
        labels = [f"{MODEL_METADATA[m][0]}\n{MODEL_METADATA[m][1].strftime('%b %d, %Y')}" for m in model_ids]
        ax.set_xticks(x_centers)
        ax.set_xticklabels(labels, rotation=0, ha="center", fontsize=9)
        ax.tick_params(axis="x", pad=10)

        # Set axis limits
        ax.set_xlim(x_base[0] - bar_width, x_base[-1] + n_features * bar_width)

        apply_clean_style(ax)
        ax.set_title(
            provider, fontsize=13, fontweight="bold",
            color=PROVIDER_COLORS_DARK[provider], pad=12,
        )

    # Set y-axis limits for all subplots
    for ax in axes:
        ax.set_ylim(-40, 1000)

    # Format y-axis ticks to show percentage values
    from matplotlib.ticker import FuncFormatter
    for ax in axes:
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{int(y)}'))

    axes[0].set_ylabel("Price change (%)", fontsize=11, color="#555555", fontweight="medium")
    axes[1].set_ylabel("")
    axes[2].set_ylabel("")

    # Main title
    title = "Price-equivalent trade-offs: All features by provider (by release date)"
    fig.suptitle(title, fontsize=15, fontweight="bold", color="#222222", y=1.02)

    # Feature legend
    from matplotlib.patches import Patch
    feature_handles = [
        Patch(facecolor="white", edgecolor="#333333", hatch=hatch, label=name)
        for name, _, _, hatch in features
    ]
    axes[2].legend(handles=feature_handles, loc="upper right", title="Feature",
                  framealpha=0.95, fontsize=9)

    # Add note
    note_text = "Positive = can raise price | Negative = must cut price"
    fig.text(
        0.5, -0.02, note_text,
        ha="center", fontsize=8, color="#666666",
    )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output_path}")


def generate_all_price_equivalent_plots(csv_path: Path, output_dir: Path) -> None:
    """Generate all price-equivalent trade-off plots."""
    # Individual feature plots
    features = [
        ("Overall Pick", "overall_pick_tag", 1.0, "overall_pick"),
        ("Rating +0.1", "rating", 0.1, "rating_increase"),
        ("Double Reviews", "log_rating_count", np.log(2), "double_reviews"),
        ("Sponsored Tag", "sponsored_tag", 1.0, "sponsored_tag"),
    ]

    for feature_name, coefficient_name, delta_z, filename in features:
        # All providers plot
        plot_single_feature_all_providers(
            csv_path,
            output_dir / f"{filename}_all_providers.png",
            feature_name=feature_name,
            coefficient_name=coefficient_name,
            delta_z=delta_z,
        )

        # By provider plot
        plot_single_feature_by_provider(
            csv_path,
            output_dir / f"{filename}_by_provider.png",
            feature_name=feature_name,
            coefficient_name=coefficient_name,
            delta_z=delta_z,
        )

    # Combined plots
    plot_combined_all_providers(csv_path, output_dir / "combined_all_providers.png")
    plot_combined_by_provider(csv_path, output_dir / "combined_by_provider.png")


if __name__ == "__main__":
    csv_path = Path("artifacts/analysis/20260122104301_choice_model.csv")
    output_dir = Path("artifacts/visualization/price_equivalent_tradeoffs")
    generate_all_price_equivalent_plots(csv_path, output_dir)
