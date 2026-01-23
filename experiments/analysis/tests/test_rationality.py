import numpy as np
import pandas as pd
import pytest

from experiments.analysis.common import SanityCheckMode
from experiments.analysis.rationality import (
    _clean_data,
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
    assert exp1_m1_q1["passed_count"] == 1

    exp1_m1_q2 = result.loc[("exp1", "m1", "q2")]
    assert exp1_m1_q2["total_experiments"] == 1
    assert exp1_m1_q2["passed_count"] == 0

    exp2_m2_q1 = result.loc[("exp2", "m2", "q1")]
    assert exp2_m2_q1["total_experiments"] == 1
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

    # exp1_m1: total=4 (3+1), passed=1 (1+0), failed=3, mean=3/4=0.75
    exp1_m1 = pooled.loc[("exp1", "m1")]
    expected_mean = 0.75
    expected_se = np.sqrt(expected_mean * (1 - expected_mean) / 4)
    assert np.isclose(exp1_m1["mean"], expected_mean)
    assert np.isclose(exp1_m1["std_error"], expected_se, rtol=1e-3)
    # CI uses Wilson method, just check they're in valid range
    assert 0 < exp1_m1["ci_lo"] < exp1_m1["mean"]
    assert exp1_m1["mean"] < exp1_m1["ci_hi"] < 1

    exp2_m2 = pooled.loc[("exp2", "m2")]
    assert exp2_m2["mean"] == 0.0
    assert exp2_m2["std_error"] == 0.0


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
    # mean=0.75, se=sqrt(0.75*0.25/4)≈0.217
    assert table.loc["Model 1", "Experiment 1"] == "0.750 (0.217)"
    assert table.loc["Model 1", "Experiment 2"] == "---"
    assert table.loc["Model 2", "Experiment 1"] == "---"
    assert table.loc["Model 2", "Experiment 2"] == "0.000 (0.000)"


def test_clean_data_filters_problematic_queries():
    rows = [
        # Target experiment + problematic queries → should be removed
        {"experiment_label": "sc_rating_increase_10_bps", "query": "washing_machine", "value": 1},
        {"experiment_label": "sc_rating_increase_10_bps", "query": "fitness_watch", "value": 2},
        # Target experiment + other query → should be kept
        {"experiment_label": "sc_rating_increase_10_bps", "query": "other_query", "value": 3},
        # Other experiment + problematic queries → should be kept
        {"experiment_label": "other_experiment", "query": "washing_machine", "value": 4},
        {"experiment_label": "other_experiment", "query": "fitness_watch", "value": 5},
    ]
    df = pd.DataFrame(rows)

    # When target experiment NOT in experiment_labels: all rows preserved
    result = _clean_data(df, experiment_labels=["other_experiment"])
    assert len(result) == 5

    # When target experiment IS in experiment_labels: filter applied
    result = _clean_data(df, experiment_labels=["sc_rating_increase_10_bps", "other_experiment"])
    assert len(result) == 3
    # Problematic queries in target experiment removed
    assert not ((result["experiment_label"] == "sc_rating_increase_10_bps") & (result["query"] == "washing_machine")).any()
    assert not ((result["experiment_label"] == "sc_rating_increase_10_bps") & (result["query"] == "fitness_watch")).any()
    # Other query in target experiment preserved
    assert ((result["experiment_label"] == "sc_rating_increase_10_bps") & (result["query"] == "other_query")).any()
    # Problematic queries in other experiments preserved
    assert ((result["experiment_label"] == "other_experiment") & (result["query"] == "washing_machine")).any()
    assert ((result["experiment_label"] == "other_experiment") & (result["query"] == "fitness_watch")).any()
