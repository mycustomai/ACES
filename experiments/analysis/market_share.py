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


def calculate_selection_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate selection statistics for products, grouped by query.

    Computes selection percentages within each query's product set.

    Args:
        df: Input dataframe with 'query', 'title', and 'selected' columns

    Returns:
        DataFrame with columns: sum, count, percentage, std_error, short_title
        indexed by (query, title)
    """
    stats = df.groupby(['query', 'title', 'model_name'])['selected'].agg(['sum', 'count'])
    stats["sum"] = stats["sum"].clip(lower=0)
    stats['percentage'] = (stats['sum'] / stats['count']) * 100

    n = stats['count']
    p = stats['sum'] / n
    stats['std_error'] = np.sqrt(p * (1 - p) / n) * 100

    return stats


