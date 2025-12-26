from pathlib import Path

import numpy as np
import pandas as pd
from experiments.analysis.common import filter_valid_experiments


def analyze_market_share(csv_path: Path, short_titles: dict, output_filepath: Path) -> None:
    """Analyze market share from experiment data and save results to CSV.

    Args:
        csv_path: Path to input CSV with experiment data
        short_titles: Nested dict mapping query -> product title -> short name
        output_filepath: Path to save the output CSV
    """
    df = pd.read_csv(csv_path)
    filtered_df = filter_valid_experiments(df)
    stats = calculate_selection_stats(filtered_df)

    # add human-readable names

    flat_short_titles: dict[tuple[str, str], str] = {
        (query, title): short_name
        for query, titles in short_titles.items()
        for title, short_name in titles.items()
    }  # (query, title) -> short_name
    stats['short_title'] = stats.index.map(flat_short_titles)

    stats.to_csv(output_filepath)


def calculate_selection_stats(df: pd.DataFrame, error_term: int = 0.001) -> pd.DataFrame:
    """Calculate selection statistics for products, grouped by query.

    Computes selection percentages within each query's product set.

    Args:
        df: Input dataframe with 'query', 'title', and 'selected' columns
        error_term: Constant to use as error term to make sure that percentage sums are sensible

    Returns:
        DataFrame with columns: sum, count, percentage, std_error, short_title
        indexed by (query, title)
    """
    df["selected"] = df["selected"].clip(lower=0)

    out = (
        df.groupby(["query", "title", "model_name"])["selected"]
        .agg(
            count="sum",
            total="size",
            mean="mean",
            sem="sem",
        )
    )
    out["std_error"] = out["sem"] * 100
    out["percentage"] = out["mean"] * 100

    out.drop(columns=["mean", "sem"], inplace=True)

    # validate percentage sums
    g = out.groupby(["query", "model_name"])
    grouped_sums = g["percentage"].sum()
    invalid = grouped_sums[(grouped_sums > 100 + error_term) | (grouped_sums < 100 - error_term)]
    assert len(invalid) == 0

    return out
