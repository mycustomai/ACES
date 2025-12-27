import numpy as np
import pandas as pd
import pytest

from experiments.analysis.common import SanityCheckMode
from experiments.analysis.rationality import (
    calculate_sanity_check,
    get_mean_and_variance,
    sanity_checks_passed,
)


@pytest.fixture
def rationality_df(tmp_path):
    rows = [
        {
            "experiment_label": "exp1",
            "model_name": "m1",
            "query": "q1",
            "experiment_number": 1,
            "selected": 1,
            "desired_choice": 1,
        },
        {
            "experiment_label": "exp1",
            "model_name": "m1",
            "query": "q1",
            "experiment_number": 1,
            "selected": 0,
            "desired_choice": 0,
        },
        {
            "experiment_label": "exp1",
            "model_name": "m1",
            "query": "q1",
            "experiment_number": 2,
            "selected": 1,
            "desired_choice": 0,
        },
        {
            "experiment_label": "exp1",
            "model_name": "m1",
            "query": "q1",
            "experiment_number": 2,
            "selected": 1,
            "desired_choice": 1,
        },
        {
            "experiment_label": "exp1",
            "model_name": "m1",
            "query": "q1",
            "experiment_number": 3,
            "selected": 0,
            "desired_choice": 1,
        },
        {
            "experiment_label": "exp1",
            "model_name": "m1",
            "query": "q1",
            "experiment_number": 3,
            "selected": 0,
            "desired_choice": 0,
        },
        {
            "experiment_label": "exp1",
            "model_name": "m1",
            "query": "q2",
            "experiment_number": 1,
            "selected": 1,
            "desired_choice": 0,
        },
        {
            "experiment_label": "exp1",
            "model_name": "m1",
            "query": "q2",
            "experiment_number": 1,
            "selected": 0,
            "desired_choice": 1,
        },
        {
            "experiment_label": "exp2",
            "model_name": "m2",
            "query": "q1",
            "experiment_number": 1,
            "selected": 1,
            "desired_choice": 1,
        },
        {
            "experiment_label": "exp3",
            "model_name": "m1",
            "query": "q1",
            "experiment_number": 1,
            "selected": 1,
            "desired_choice": 1,
        },
    ]
    path = tmp_path / "rationality.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return pd.read_csv(path)


def test_sanity_checks_passed_counts_and_filtering(rationality_df):
    result = sanity_checks_passed(
        rationality_df,
        experiment_labels=["exp1", "exp2"],
        models=["m1", "m2"],
        queries=["q1", "q2"],
    )

    exp1_m1_q1 = result.loc[("exp1", "m1", "q1")]
    assert exp1_m1_q1["total_experiments"] == 3
    assert exp1_m1_q1["experiments_without_error"] == 2
    assert exp1_m1_q1["passed_count"] == 2

    exp1_m1_q2 = result.loc[("exp1", "m1", "q2")]
    assert exp1_m1_q2["total_experiments"] == 1
    assert exp1_m1_q2["experiments_without_error"] == 1
    assert exp1_m1_q2["passed_count"] == 0

    exp2_m2_q1 = result.loc[("exp2", "m2", "q1")]
    assert exp2_m2_q1["total_experiments"] == 1
    assert exp2_m2_q1["experiments_without_error"] == 1
    assert exp2_m2_q1["passed_count"] == 1

    assert ("exp3", "m1", "q1") not in result.index


def test_get_mean_and_variance_pooled_stats(rationality_df):
    success = sanity_checks_passed(
        rationality_df,
        experiment_labels=["exp1", "exp2"],
        models=["m1", "m2"],
        queries=["q1", "q2"],
    )
    pooled = get_mean_and_variance(success)

    exp1_m1 = pooled.loc[("exp1", "m1")]
    expected_mean = 1 / 3
    expected_se = 1 / 3
    expected_ci_lo = expected_mean - 1.96 * expected_se
    expected_ci_hi = expected_mean + 1.96 * expected_se
    assert np.isclose(exp1_m1["mean"], expected_mean)
    assert np.isclose(exp1_m1["std_error"], expected_se, rtol=1e-3)
    assert np.isclose(exp1_m1["ci_lo"], expected_ci_lo, rtol=1e-3)
    assert np.isclose(exp1_m1["ci_hi"], expected_ci_hi, rtol=1e-3)

    exp2_m2 = pooled.loc[("exp2", "m2")]
    assert exp2_m2["mean"] == 0.0
    assert exp2_m2["std_error"] == 0.0
    assert np.isnan(exp2_m2["ci_lo"])
    assert np.isnan(exp2_m2["ci_hi"])


def test_calculate_sanity_check_formats_and_orders(rationality_df):
    experiment_names = {
        SanityCheckMode.PRICE: {"exp1": "Experiment 1", "exp2": "Experiment 2"}
    }
    model_display_names = {"m1": "Model 1", "m2": "Model 2"}

    table = calculate_sanity_check(
        rationality_df,
        SanityCheckMode.PRICE,
        experiment_names,
        model_display_names,
    )

    assert list(table.columns) == ["Experiment 1", "Experiment 2"]
    assert table.index.name == "Model"
    assert table.loc["Model 1", "Experiment 1"] == "0.333 (0.333)"
    assert table.loc["Model 1", "Experiment 2"] == "---"
    assert table.loc["Model 2", "Experiment 1"] == "---"
    assert table.loc["Model 2", "Experiment 2"] == "0.000 (0.000)"
