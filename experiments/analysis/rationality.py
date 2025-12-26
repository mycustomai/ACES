from collections import defaultdict
from typing import TypedDict, Dict, Iterable

import numpy as np
import pandas as pd
import scipy.stats as stats

from experiments.analysis.common import SanityCheckMode


class ExperimentSanityCheckResults(TypedDict):
    total_experiments: int
    experiments_without_error: int
    passed_count: int


type ExperimentLabelKey = str
type ModelNameKey = str
type QueryKey = str
type SanityCheckDict = Dict[ExperimentLabelKey, Dict[ModelNameKey, Dict[QueryKey, ExperimentSanityCheckResults]]]


def sanity_checks_passed(df: pd.DataFrame, experiment_labels: Iterable[str], models: list[str], queries: list[str]) -> SanityCheckDict:
    output = defaultdict(lambda: defaultdict(dict))
    # TODO: use `groupby` and iterate AND/OR vectorize
    for experiment_label in experiment_labels:
        for model in models:
            for query in queries:
                df_filtered = df[(df["model_name"] == model) & (df["experiment_label"] == experiment_label) & (df["query"] == query)]
                experiment_numbers = sorted(df_filtered["experiment_number"].unique())
                experiments_without_error = 0
                passed_count = 0
                for experiment_number in experiment_numbers:
                    df_filtered_exp = df_filtered[df_filtered["experiment_number"] == experiment_number]
                    if df_filtered_exp["selected"].sum() > 0:
                        experiments_without_error += 1
                        # TODO: simplify
                        if (df_filtered_exp["selected"] == df_filtered_exp["desired_choice"]).all() or df_filtered_exp["selected"].sum() == 2:
                            passed_count += 1
                        else:
                            continue
                if len(experiment_numbers) > 0:
                    output[experiment_label][model][query] = {
                        "total_experiments": len(experiment_numbers),
                        "experiments_without_error": experiments_without_error,
                        "passed_count": passed_count,
                    }
    return output


def get_mean_and_variance(data: SanityCheckDict) -> dict:
    meta_results = defaultdict(dict)
    for experiment_label, models in data.items():
        for model, queries in models.items():
            successes = []
            trials = []
            for query, results in queries.items():
                successes.append(results["passed_count"])
                trials.append(results["experiments_without_error"])

            successes = np.array(successes)
            trials = np.array(trials)

            props = 1 - successes / trials
            variances = props * (1 - props) / trials

            n = trials.sum()
            pooled_mu = (trials * props).sum() / n
            pooled_var = (((trials - 1) * variances) + trials * (props - pooled_mu)**2).sum() / (n - 1)

            ci = stats.norm.interval(0.95, loc=pooled_mu, scale=np.sqrt(pooled_var / n))
            meta_results[experiment_label][model] = (pooled_mu, (ci[0], ci[1]))

    return meta_results


def calculate_sanity_check(df: pd.DataFrame, check_name: SanityCheckMode, experiment_names: dict[str, dict[str, str]], model_display_names: dict[str, str]) -> pd.DataFrame:
    selected_experiment_names = experiment_names[check_name]

    models = df["model_name"].unique().tolist()
    queries = df["query"].unique().tolist()
    success_data = sanity_checks_passed(df=df, experiment_labels=selected_experiment_names.keys(), models=models, queries=queries)
    meta_data = get_mean_and_variance(success_data)

    table_data = []
    for model in models:
        row = {"Model": model_display_names.get(model, model)}
        for experiment_label in selected_experiment_names.keys():
            if model in meta_data[experiment_label]:
                mean, (ci_lo, ci_hi) = meta_data[experiment_label][model]

                # calculate std err from confidence interval
                # for 95% CI: mean ± 1.96*SE, so SE = (ci_hi - ci_lo) / (2 * 1.96)
                se = (ci_hi - ci_lo) / (2 * 1.96)

                # handle NaN
                if np.isnan(se):
                    row[selected_experiment_names[experiment_label]] = f"{mean:.3f} (0.000)"
                else:
                    row[selected_experiment_names[experiment_label]] = f"{mean:.3f} ({se:.3f})"
            else:
                row[selected_experiment_names[experiment_label]] = "---"
        table_data.append(row)

    results_table = pd.DataFrame(table_data)
    results_table.set_index("Model", inplace=True)

    return results_table
