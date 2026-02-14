from datetime import date

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

# Modern, accessible font and styling
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["SF Pro Display", "Segoe UI", "Helvetica Neue", "Arial"],
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
    "axes.facecolor": "#FAFAFA",
    "figure.facecolor": "white",
})

# (display_name, release_date, provider)
MODEL_METADATA = {
    "gpt-4o-2024-11-20": ("GPT-4o", date(2024, 11, 20), "OpenAI"),
    "gemini-2.0-flash-001": ("Gemini 2.0 Flash", date(2025, 2, 5), "Google"),
    "claude-opus-4-20250514": ("Claude Opus 4", date(2025, 5, 22), "Anthropic"),
    "gemini-2.5-pro": ("Gemini 2.5 Pro", date(2025, 6, 17), "Google"),
    "claude-opus-4-1-20250805": ("Claude Opus 4.1", date(2025, 8, 5), "Anthropic"),
    "gpt-5": ("GPT-5", date(2025, 8, 7), "OpenAI"),
    "claude-sonnet-4-5-20250929": ("Claude Sonnet 4.5", date(2025, 9, 29), "Anthropic"),
    "claude-opus-4-5-20251101": ("Claude Opus 4.5", date(2025, 11, 24), "Anthropic"),
    "gpt-5.1": ("GPT-5.1", date(2025, 11, 12), "OpenAI"),
    "gemini-3-pro-preview": ("Gemini 3 Pro", date(2025, 11, 18), "Google"),
    "gemini-3-flash-preview": ("Gemini 3 Flash", date(2025, 12, 17), "Google"),
    "gpt-5.2": ("GPT-5.2", date(2025, 12, 11), "OpenAI"),
}

# display name → (model_id, release_date, provider)
DISPLAY_NAME_TO_METADATA = {
    v[0]: (k, v[1], v[2]) for k, v in MODEL_METADATA.items()
}
# also handle slight naming differences in sanity check CSVs
DISPLAY_NAME_TO_METADATA["Gemini 3 Flash Preview"] = DISPLAY_NAME_TO_METADATA["Gemini 3 Flash"]
DISPLAY_NAME_TO_METADATA["Gemini 3 Pro Preview"] = DISPLAY_NAME_TO_METADATA["Gemini 3 Pro"]

PROVIDER_COLORS = {
    "Anthropic": "#D97757",  # Warm coral-orange (official brand color)
    "Google": "#4285F4",     # Google blue (official brand color)
    "OpenAI": "#000000",     # Black (official brand color)
}

# Lighter variants for gradients and highlights
PROVIDER_COLORS_LIGHT = {
    "Anthropic": "#F4B8A0",
    "Google": "#8AB4F8",
    "OpenAI": "#666666",  # Gray for lighter variant
}

# Darker variants for emphasis
PROVIDER_COLORS_DARK = {
    "Anthropic": "#B85A3A",
    "Google": "#1967D2",
    "OpenAI": "#000000",  # Black stays black
}


def apply_clean_style(ax: plt.Axes, use_percent: bool = True) -> None:
    """Apply modern, clean styling to axes."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#E0E0E0")
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_color("#E0E0E0")
    ax.spines["bottom"].set_linewidth(1.5)
    ax.tick_params(colors="#666666", labelsize=10, width=1.2)
    if use_percent:
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#E8E8E8", linewidth=1.0, linestyle="-", alpha=0.7)
    ax.xaxis.grid(False)
    ax.set_facecolor("#FAFAFA")


def sort_display_names_by_date(names: list[str]) -> list[str]:
    return sorted(names, key=lambda n: DISPLAY_NAME_TO_METADATA[n][1])


def get_provider_for_display_name(name: str) -> str:
    return DISPLAY_NAME_TO_METADATA[name][2]


def draw_rounded_bar(
    ax: plt.Axes,
    x: float,
    height: float,
    width: float,
    color: str,
    fill: str = "white",
    rounding: float = 0.012,
    hatch: str | None = "///",
) -> None:
    """Draw a bar with hatched pattern, rounded only on top."""
    from matplotlib.patches import Rectangle
    from matplotlib.path import Path
    from matplotlib.patches import PathPatch

    x_left = x - width / 2
    x_right = x + width / 2
    corner_radius = rounding * 10  # Convert to absolute size

    # Use simple rectangle with rounded top if matplotlib supports it
    # Otherwise, just use FancyBboxPatch with custom style
    rect = FancyBboxPatch(
        (x_left, 0),
        width, height,
        boxstyle=f"round,pad=0,rounding_size={rounding}",
        facecolor=fill,
        edgecolor=color,
        linewidth=1.0,
        hatch=hatch,
    )
    ax.add_patch(rect)


def add_value_label(
    ax: plt.Axes,
    x: float,
    y: float,
    value: float,
    color: str = "#333333",
    fontsize: int = 8,
    format_str: str = "{:.1%}",
    offset_y: float = 0.005,
    rotation: int = 0,
    skip_zero: bool = True,
) -> None:
    """Add a value label above a bar."""
    threshold = 0.001 if skip_zero else -1  # Skip values less than 0.1% if skip_zero is True
    if not np.isnan(value) and value > threshold:
        ax.text(
            x, y + offset_y,
            format_str.format(value),
            ha="center", va="bottom",
            fontsize=fontsize, color=color,
            fontweight="medium",
            rotation=rotation,
        )


def create_modern_legend(
    ax: plt.Axes,
    provider_colors: dict,
    title: str = "Provider",
    loc: str = "upper right",
) -> None:
    """Create a modern, clean legend."""
    from matplotlib.patches import Patch

    handles = [
        Patch(
            facecolor=provider_colors[p],
            edgecolor=PROVIDER_COLORS_DARK.get(p, provider_colors[p]),
            linewidth=1.5,
            label=p,
            alpha=0.8,
        )
        for p in provider_colors
    ]
    legend = ax.legend(
        handles=handles,
        title=title,
        loc=loc,
        fontsize=9,
        title_fontsize=10,
        frameon=True,
        fancybox=False,
        edgecolor="#D0D0D0",
        framealpha=0.95,
        borderpad=0.8,
        labelspacing=0.6,
    )
    legend.get_frame().set_facecolor("white")
    legend.get_title().set_fontweight("semibold")
    legend.get_title().set_color("#333333")
