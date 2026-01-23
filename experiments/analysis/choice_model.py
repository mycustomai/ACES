from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from patsy import dmatrices

from experiments.analysis.common import filter_valid_experiments



def _filter_experiments(df: pd.DataFrame, min_selection_rate: float = 0.0) -> pd.DataFrame:
    """
    Filter experiments and remove products with low selection rates.

    Args:
        df: Input dataframe with experiment data
        min_selection_rate: Minimum selection rate threshold. Products with selection
                           rate <= this value will be removed. Default 0.0 removes
                           only never-selected products.
    """
    filtered_df = filter_valid_experiments(df)

    # convert selected column to binary (0 or 1)
    filtered_df["selected"] = (filtered_df["selected"] > 0).astype(int)

    # filter ignored queries
    sel_queries = filtered_df["query"].unique()
    idx = np.where(sel_queries == "usb_cable")
    sel_queries = np.delete(sel_queries, idx[0])

    filtered_df = filtered_df[filtered_df["query"].isin(sel_queries)]

    # drop products with low selection rates (per model)
    title_selection_rates = filtered_df.groupby(["model_name", "title"])["selected"].mean()
    pairs_to_remove = title_selection_rates[title_selection_rates <= min_selection_rate].index.tolist()
    mask = filtered_df.set_index(["model_name", "title"]).index.isin(pairs_to_remove)
    filtered_df = filtered_df[~mask]

    return filtered_df


def _prepare_choice_selection_df(df: pd.DataFrame) -> pd.DataFrame:
    # convert to row/columns
    df["row"] = df["assigned_position"] // 4 + 1
    df["column"] = df["assigned_position"] % 4 + 1

    df["log_price"] = np.log(df["price"])
    df["log_rating_count"] = np.log(df["rating_count"] + 1)

    df["choice_set_id"] = df["query"].astype(str) + "_" + df["experiment_label"].astype(str) + "_" + df["experiment_number"].astype(str)


    df["row_1_dummy"] = (df["row"] == 1).astype(int)
    df["col_1_dummy"] = (df["column"] == 1).astype(int)
    df["col_2_dummy"] = (df["column"] == 2).astype(int)
    df["col_3_dummy"] = (df["column"] == 3).astype(int)

    df["sponsored_tag"] = (df["sponsored"] == True).astype(int)
    df["overall_pick_tag"] = (df["overall_pick"] == True).astype(int)
    df["scarcity_tag"] = (df["low_stock"] == True).astype(int)

    return df


MODEL_FORMULA = (
    "selected ~ -1 + row_1_dummy + col_1_dummy + col_2_dummy + col_3_dummy + "
    "log_price + log_rating_count + rating + sponsored_tag + overall_pick_tag + "
    "scarcity_tag + C(title)"
)


def _fit_single_model(group_df: pd.DataFrame) -> pd.Series:
    """
    Fit choice model for a single model.

    Args:
        group_df: Prepared dataframe for this model

    Returns:
        Series with coefficients and fit statistics

    Raises:
        Exception if model fitting fails
    """
    y, x = dmatrices(MODEL_FORMULA, data=group_df, return_type="dataframe")
    model = sm.ConditionalLogit(y, x, groups=group_df["choice_set_id"])
    result = model.fit(method="newton", disp=False)

    # Filter out title coefficients
    params = result.params[~result.params.index.str.startswith("C(title)")].copy()
    llf = result.llf
    llnull = model.loglike(np.zeros(len(result.params)))
    params["llf"] = llf
    params["llnull"] = llnull
    params["pseudo_r2"] = 1 - llf / llnull

    return params


def fit_choice_model(df: pd.DataFrame) -> pd.DataFrame:
    """Fit choice model for all models in the dataframe."""
    coefficients = {}
    for model_name, group_df in df.groupby("model_name"):
        params = _fit_single_model(group_df)
        coefficients[model_name] = params

    return pd.DataFrame(coefficients)


def generate_choice_model_results(input_csv: Path, output_filepath: Path) -> None:
    """
    Generate choice model results with automatic retry for models that fail.

    For models that encounter numerical issues (e.g., due to products with very low
    selection rates), the function will progressively increase the minimum selection
    rate threshold until fitting succeeds.
    """
    df = pd.read_csv(input_csv)

    # Threshold progression for retries
    selection_rate_thresholds = [0.0, 0.001, 0.002, 0.005, 0.01]

    coefficients = {}

    for model_name in df["model_name"].unique():
        model_df = df[df["model_name"] == model_name]
        model_succeeded = False

        for threshold in selection_rate_thresholds:
            try:
                filtered_df = _filter_experiments(model_df, min_selection_rate=threshold)
                prepared_df = _prepare_choice_selection_df(filtered_df.copy())
                params = _fit_single_model(prepared_df)

                # Check if fitting produced valid coefficients (not NaN)
                if params.isna().any():
                    raise ValueError("Model fitting produced NaN coefficients")

                params["min_selection_rate_used"] = threshold

                if threshold > 0:
                    print(f"{model_name}: succeeded with min_selection_rate={threshold}")

                coefficients[model_name] = params
                model_succeeded = True
                break  # Success, move to next model

            except Exception as e:
                print(f"{model_name}: failed with threshold={threshold}: {e}")
                continue  # Try next threshold

        if not model_succeeded:
            print(f"{model_name}: FAILED with all thresholds")

    choice_model_results = pd.DataFrame(coefficients)
    choice_model_results = choice_model_results.reindex(sorted(choice_model_results.columns), axis=1)
    choice_model_results.to_csv(output_filepath)