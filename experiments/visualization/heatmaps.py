from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

from experiments.visualization.common import (
    MODEL_METADATA,
    PROVIDER_COLORS,
    PROVIDER_COLORS_DARK,
)


def compute_position_probabilities(row1_coef: float, col1_coef: float, col2_coef: float, col3_coef: float) -> np.ndarray:
    """
    Compute selection probabilities for each position in a 2x4 grid.

    Args:
        row1_coef: Coefficient for row 1 (row 2 is baseline = 0)
        col1_coef: Coefficient for column 1 (col 4 is baseline = 0)
        col2_coef: Coefficient for column 2
        col3_coef: Coefficient for column 3

    Returns:
        2x4 numpy array of probabilities
    """
    # Position utilities: row + col coefficients
    # Row 1, Cols 1-4: row1 + col1, row1 + col2, row1 + col3, row1 + 0
    # Row 2, Cols 1-4: 0 + col1, 0 + col2, 0 + col3, 0 + 0
    utilities = np.array([
        [row1_coef + col1_coef, row1_coef + col2_coef, row1_coef + col3_coef, row1_coef],
        [col1_coef, col2_coef, col3_coef, 0.0],
    ])

    # Softmax to get probabilities
    exp_utilities = np.exp(utilities)
    probabilities = exp_utilities / np.sum(exp_utilities)

    return probabilities


def plot_heatmap(
    model_id: str,
    probabilities: np.ndarray,
    output_path: Path,
    vmin: float = 0.05,
    vmax: float = 0.25,
    hide_colorbar: bool = True,
) -> None:
    """
    Create a heatmap visualization of position probabilities.

    Args:
        model_id: Model identifier
        probabilities: 2x4 numpy array of probabilities
        output_path: Path to save the figure
        vmin: Minimum value for color scale
        vmax: Maximum value for color scale
        hide_colorbar: Whether to hide the colorbar
    """
    provider = MODEL_METADATA[model_id][2]
    display_name = MODEL_METADATA[model_id][0]

    num_rows, num_cols = probabilities.shape

    fig, ax = plt.subplots(figsize=(8, 4))

    # Use YlGnBu colormap like the example
    cmap = plt.cm.YlGnBu
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    # Define box size and gap
    box_width = 1.0
    box_height = 1.0
    gap_x = 0.1
    gap_y = 0.1

    # Draw each box
    for r in range(num_rows):
        for c in range(num_cols):
            prob_value = probabilities[r, c]
            color = cmap(norm(prob_value))

            # Calculate position for the rectangle
            x_start = c * (box_width + gap_x)
            y_start = (num_rows - 1 - r) * (box_height + gap_y)  # Invert y-axis

            rect = plt.Rectangle(
                (x_start, y_start), box_width, box_height,
                facecolor=color, edgecolor='black', linewidth=1.5,
                clip_on=False
            )
            ax.add_patch(rect)

            # Add text label for probability
            # Determine text color based on background brightness
            luminance = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
            text_color = 'black' if luminance > 0.5 else 'white'

            ax.text(
                x_start + box_width / 2, y_start + box_height / 2,
                f'{prob_value:.1%}',
                ha='center', va='center', color=text_color, fontsize=32,
                fontfamily='Helvetica'
            )

    # Set limits and hide axes
    ax.set_xlim(-gap_x/2, num_cols * (box_width + gap_x))
    ax.set_ylim(-gap_y/2, num_rows * (box_height + gap_y))
    ax.axis('off')

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white', transparent=False)
    plt.close(fig)
    print(f"Saved: {output_path}")


def generate_all_heatmaps(csv_path: Path, output_dir: Path) -> None:
    """
    Generate heatmaps for all models in the choice model CSV.

    Args:
        csv_path: Path to choice model coefficients CSV
        output_dir: Directory to save heatmaps
    """
    # Load choice model coefficients
    df = pd.read_csv(csv_path, index_col=0)

    # Get all models that are in MODEL_METADATA
    model_ids = [m for m in df.columns if m in MODEL_METADATA]

    # Determine vmin and vmax across all models for consistent color scale
    all_probs = []
    for model_id in model_ids:
        row1 = df.loc["row_1_dummy", model_id]
        col1 = df.loc["col_1_dummy", model_id]
        col2 = df.loc["col_2_dummy", model_id]
        col3 = df.loc["col_3_dummy", model_id]

        probs = compute_position_probabilities(row1, col1, col2, col3)
        all_probs.extend(probs.flatten())

    vmin = max(0.0, min(all_probs) - 0.01)  # Add small margin
    vmax = min(1.0, max(all_probs) + 0.01)  # Add small margin

    # Generate heatmap for each model
    for model_id in model_ids:
        row1 = df.loc["row_1_dummy", model_id]
        col1 = df.loc["col_1_dummy", model_id]
        col2 = df.loc["col_2_dummy", model_id]
        col3 = df.loc["col_3_dummy", model_id]

        probs = compute_position_probabilities(row1, col1, col2, col3)

        # Use model_id as filename (safe for filesystem)
        output_path = output_dir / f"heatmap_{model_id}.png"

        plot_heatmap(
            model_id=model_id,
            probabilities=probs,
            output_path=output_path,
            vmin=vmin,
            vmax=vmax,
            hide_colorbar=True,
        )


if __name__ == "__main__":
    csv_path = Path("artifacts/analysis/20260122104301_choice_model.csv")
    output_dir = Path("artifacts/visualization/heatmaps")

    generate_all_heatmaps(csv_path, output_dir)
