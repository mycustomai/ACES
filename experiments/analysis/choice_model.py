from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from patsy import dmatrices

from experiments.analysis.common import filter_valid_experiments



def _filter_experiments(df: pd.DataFrame) -> pd.DataFrame:
    filtered_df = filter_valid_experiments(df)

    # filter ignored queries
    sel_queries = filtered_df["query"].unique()
    idx = np.where(sel_queries == "usb_cable")
    sel_queries = np.delete(sel_queries, idx[0])

    filtered_df = filtered_df[filtered_df["query"].isin(sel_queries)]

    # drop unselected products (per model)
    title_selection_rates = filtered_df.groupby(["model_name", "title"])["selected"].mean()
    pairs_to_remove = title_selection_rates[title_selection_rates == 0.0].index.tolist()
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


def fit_choice_model(df: pd.DataFrame) -> pd.DataFrame:
    model_formula = (
        "selected ~ -1 + row_1_dummy + col_1_dummy + col_2_dummy + col_3_dummy + "
        "log_price + log_rating_count + rating + sponsored_tag + overall_pick_tag + "
        "scarcity_tag + C(title)"
    )
    coefficients = {}
    for model_name, group_df in df.groupby("model_name"):
        y, x = dmatrices(model_formula, data=group_df, return_type="dataframe")

        # Diagnostic: Check unique values in selected
        print(f"Model: {model_name}")
        print(f"Unique values in y: {y['selected'].unique()}")
        print(f"NaN count in y: {y['selected'].isna().sum()}")

        model = sm.ConditionalLogit(y, x, groups=group_df["choice_set_id"])
        result = model.fit(method="newton")

        params = result.params.copy()
        llf = result.llf
        llnull = model.loglike(np.zeros(len(result.params)))
        params["llf"] = llf
        params["llnull"] = llnull
        params["pseudo_r2"] = 1 - llf / llnull
        coefficients[model_name] = params

    return pd.DataFrame(coefficients)


def generate_choice_model_results(input_csv: Path, output_filepath: Path) -> None:
    df = pd.read_csv(input_csv)

    filtered_df = _filter_experiments(df)
    prepared_df = _prepare_choice_selection_df(filtered_df)
    choice_model_results = fit_choice_model(prepared_df)

    choice_model_results.to_csv(output_filepath)