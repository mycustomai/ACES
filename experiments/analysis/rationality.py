from typing import Iterable

import numpy as np
import pandas as pd
import statsmodels.api as sm
import scipy.stats as stats

from experiments.analysis.common import SanityCheckMode


def sanity_checks_passed(
    orig_df: pd.DataFrame,
    experiment_labels: Iterable[str],
    models: list[str],
    queries: list[str],
) -> pd.DataFrame:
    mask = (
        orig_df["experiment_label"].isin(experiment_labels)
        & orig_df["model_name"].isin(models)
        & orig_df["query"].isin(queries)
    )
    df_filtered = orig_df.loc[mask].copy()

    df_filtered["selected"] = df_filtered["selected"].clip(lower=0, upper=1)

    def experiment_passed(g: pd.DataFrame) -> pd.Series:
        passed = (
            (g["selected"] == g["desired_choice"]).all()
        )
        return pd.Series({"passed": passed, "total": len(g)})

    exp_results = df_filtered.groupby(
        ["experiment_label", "model_name", "query", "experiment_number"], as_index=False
    ).apply(experiment_passed, include_groups=False)

    return exp_results.groupby(["experiment_label", "model_name", "query"]).agg(
        total_experiments=("total", "count"),
        passed_count=("passed", "sum"),
    )


def get_mean_and_variance(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()

    def compute_group_stats(g: pd.DataFrame) -> pd.Series:
        total = g["total_experiments"].sum()
        passed = g["passed_count"].sum()
        failed = total - passed

        mean = failed / total  # proportion failed

        ci = sm.stats.proportion_confint(
            count=failed, nobs=total, alpha=0.05, method="wilson"
        )
        se = np.sqrt(mean * (1 - mean) / total)

        return pd.Series({
            "mean": mean,
            "ci_lo": ci[0],
            "ci_hi": ci[1],
            "std_error": se,
        })

    return df.groupby(["experiment_label", "model_name"]).apply(
        compute_group_stats, include_groups=False
    )


def calculate_sanity_check(
    df: pd.DataFrame,
    check_name: SanityCheckMode,
    experiment_names: dict[str, dict[str, str]],
    model_display_names: dict[str, str],
) -> pd.DataFrame:
    selected_experiment_names = experiment_names[check_name]
    models = df["model_name"].unique().tolist()
    queries = df["query"].unique().tolist()

    success_data = sanity_checks_passed(
        df, experiment_labels=selected_experiment_names.keys(), models=models, queries=queries
    )

    meta_data = get_mean_and_variance(success_data).reset_index()

    meta_data["cell"] = meta_data.apply(
        lambda r: f"{r['mean']:.3f} ({r['std_error']:.3f})", axis=1
    )
    meta_data["display_exp"] = meta_data["experiment_label"].map(selected_experiment_names)

    table = meta_data.pivot(index="model_name", columns="display_exp", values="cell")
    table.index = table.index.map(lambda m: model_display_names.get(m, m))
    table.index.name = "Model"

    return table.reindex(columns=selected_experiment_names.values()).fillna("---")
