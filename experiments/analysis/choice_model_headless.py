"""
Choice model for headless experiments.

This module fits a conditional logit choice model for headless experiments,
where products are presented as a linear list (positions 1-8) rather than
a 2x4 grid layout used in screenshot experiments.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from patsy import dmatrices


def filter_valid_experiments_headless(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to experiments where exactly one product was chosen.

    Args:
        df: DataFrame with headless experiment results

    Returns:
        Filtered DataFrame with only valid experiments
    """
    valid_experiments = df.groupby(["model", "query", "experiment_number"]).filter(
        lambda x: x["chosen"].sum() == 1
    )
    return valid_experiments


def _filter_experiments(df: pd.DataFrame) -> pd.DataFrame:
    """Filter and clean headless experiment data.

    Args:
        df: Raw DataFrame from headless experiment CSV

    Returns:
        Cleaned and filtered DataFrame
    """
    filtered_df = filter_valid_experiments_headless(df)

    # Filter ignored queries
    sel_queries = filtered_df["query"].unique()
    idx = np.where(sel_queries == "usb_cable")
    sel_queries = np.delete(sel_queries, idx[0])
    filtered_df = filtered_df[filtered_df["query"].isin(sel_queries)]

    # Drop products that were never selected (per model)
    title_selection_rates = filtered_df.groupby(["model", "title"])["chosen"].mean()
    pairs_to_remove = title_selection_rates[title_selection_rates == 0.0].index.tolist()
    mask = filtered_df.set_index(["model", "title"]).index.isin(pairs_to_remove)
    filtered_df = filtered_df[~mask]

    return filtered_df


def _prepare_choice_selection_df(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare DataFrame for choice model fitting.

    Creates position dummies for linear list layout (positions 0-7),
    log transforms for price and review count, and tag dummies.

    Args:
        df: Filtered DataFrame

    Returns:
        DataFrame with additional columns for model fitting
    """
    df = df.copy()

    # Log transforms
    df["log_price"] = np.log(df["price"])
    df["log_rating_count"] = np.log(df["number_of_reviews"] + 1)

    # Choice set ID (no experiment_label in headless experiments)
    df["choice_set_id"] = (
        df["query"].astype(str) + "_" + df["experiment_number"].astype(str)
    )

    # Position dummies for linear list (positions 0-7)
    for pos in range(8):
        df[f"position_{pos + 1}_dummy"] = (df["position_in_experiment"] == pos).astype(int)

    # Tag dummies
    df["sponsored_tag"] = (df["sponsored"] == True).astype(int)
    df["overall_pick_tag"] = (df["overall_pick_tag"] == True).astype(int)
    df["scarcity_tag"] = (df["scarcity_tag"] == True).astype(int)

    return df


def fit_choice_model(df: pd.DataFrame) -> pd.DataFrame:
    """Fit conditional logit choice model for each model.

    Uses position dummies (1-7, with position 8 as reference),
    log price, log rating count, rating, and tag indicators.

    Args:
        df: Prepared DataFrame with all necessary columns

    Returns:
        DataFrame with coefficients for each model
    """
    # Position dummies 1-7 (position 8 is reference category)
    position_terms = " + ".join([f"position_{i}_dummy" for i in range(1, 8)])

    model_formula = (
        f"chosen ~ -1 + {position_terms} + "
        "log_price + log_rating_count + rating + sponsored_tag + overall_pick_tag + "
        "scarcity_tag + C(title)"
    )

    coefficients = {}
    for model_name, group_df in df.groupby("model"):
        try:
            y, x = dmatrices(model_formula, data=group_df, return_type="dataframe")
            model = sm.ConditionalLogit(y, x, groups=group_df["choice_set_id"])
            result = model.fit(method="newton")

            # Filter out title coefficients
            params = result.params[~result.params.index.str.startswith("C(title)")].copy()

            # Calculate pseudo R-squared
            llf = result.llf
            llnull = model.loglike(np.zeros(len(result.params)))
            params["llf"] = llf
            params["llnull"] = llnull
            params["pseudo_r2"] = 1 - llf / llnull

            coefficients[model_name] = params
        except Exception as e:
            print(f"Error fitting model for {model_name}: {e}")
            continue

    return pd.DataFrame(coefficients)


def generate_choice_model_results(input_csv: Path, output_filepath: Path) -> None:
    """Generate choice model results from headless experiment CSV.

    Args:
        input_csv: Path to input CSV file with headless experiment results
        output_filepath: Path to save the choice model coefficients
    """
    df = pd.read_csv(input_csv)

    filtered_df = _filter_experiments(df)
    prepared_df = _prepare_choice_selection_df(filtered_df)
    choice_model_results = fit_choice_model(prepared_df)

    choice_model_results.to_csv(output_filepath)
    print(f"Choice model results saved to {output_filepath}")
