import numpy as np
import pandas as pd
import pytest

from experiments.analysis.market_share import analyze_market_share, calculate_selection_stats


@pytest.fixture
def selection_stats_df(tmp_path):
    rows = [
        {"query": "q1", "title": "A", "model_name": "m1", "selected": -1},
        {"query": "q1", "title": "A", "model_name": "m1", "selected": -1},
        {"query": "q1", "title": "B", "model_name": "m1", "selected": 1},
        {"query": "q1", "title": "B", "model_name": "m1", "selected": 0},
        {"query": "q1", "title": "B", "model_name": "m1", "selected": 1},
        {"query": "q2", "title": "C", "model_name": "m2", "selected": 1},
    ]
    path = tmp_path / "market_share_stats.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return pd.read_csv(path)


@pytest.fixture
def market_share_csv(tmp_path):
    rows = [
        {
            "query": "q1",
            "title": "A",
            "model_name": "m1",
            "experiment_label": "expA",
            "experiment_number": 1,
            "selected": 1,
        },
        {
            "query": "q1",
            "title": "B",
            "model_name": "m1",
            "experiment_label": "expA",
            "experiment_number": 1,
            "selected": 0,
        },
        {
            "query": "q1",
            "title": "A",
            "model_name": "m1",
            "experiment_label": "expA",
            "experiment_number": 2,
            "selected": 0,
        },
        {
            "query": "q1",
            "title": "B",
            "model_name": "m1",
            "experiment_label": "expA",
            "experiment_number": 2,
            "selected": 0,
        },
        {
            "query": "q2",
            "title": "C",
            "model_name": "m2",
            "experiment_label": "expA",
            "experiment_number": 1,
            "selected": -1,
        },
        {
            "query": "q2",
            "title": "D",
            "model_name": "m2",
            "experiment_label": "expA",
            "experiment_number": 1,
            "selected": 0,
        },
    ]
    path = tmp_path / "market_share.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_calculate_selection_stats_clips_and_computes(selection_stats_df):
    stats = calculate_selection_stats(selection_stats_df)

    q1_a = stats.loc[("q1", "A", "m1")]
    assert q1_a["sum"] == 0
    assert q1_a["count"] == 2
    assert q1_a["percentage"] == 0
    assert q1_a["std_error"] == 0

    q1_b = stats.loc[("q1", "B", "m1")]
    expected_percentage = 2 / 3 * 100
    expected_std_error = np.sqrt((2 / 3) * (1 - 2 / 3) / 3) * 100
    assert q1_b["sum"] == 2
    assert q1_b["count"] == 3
    assert np.isclose(q1_b["percentage"], expected_percentage)
    assert np.isclose(q1_b["std_error"], expected_std_error)

    q2_c = stats.loc[("q2", "C", "m2")]
    assert q2_c["sum"] == 1
    assert q2_c["count"] == 1
    assert q2_c["percentage"] == 100
    assert q2_c["std_error"] == 0


def test_analyze_market_share_filters_and_writes_csv(market_share_csv, tmp_path):
    output_path = tmp_path / "market_share_output.csv"
    short_titles = {
        "q1": {"A": "Short A", "B": "Short B"},
        "q2": {"C": "Short C", "D": "Short D"},
    }

    analyze_market_share(market_share_csv, short_titles, output_path)

    output = pd.read_csv(output_path)
    assert set(output[["query", "title", "model_name"]].itertuples(index=False, name=None)) == {
        ("q1", "A", "m1"),
        ("q1", "B", "m1"),
    }

    row_a = output[
        (output["query"] == "q1")
        & (output["title"] == "A")
        & (output["model_name"] == "m1")
    ].iloc[0]
    row_b = output[
        (output["query"] == "q1")
        & (output["title"] == "B")
        & (output["model_name"] == "m1")
    ].iloc[0]
    assert row_a["sum"] == 1
    assert row_a["count"] == 1
    assert np.isclose(row_a["percentage"], 100.0)
    assert np.isclose(row_a["std_error"], 0.0)
    assert row_b["sum"] == 0
    assert row_b["count"] == 1
    assert np.isclose(row_b["percentage"], 0.0)
    assert np.isclose(row_b["std_error"], 0.0)
    assert "short_title" in output.columns
    assert row_a["short_title"] == "Short A"
    assert row_b["short_title"] == "Short B"
