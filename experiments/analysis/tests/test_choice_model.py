import numpy as np
import pandas as pd

from experiments.analysis.choice_model import generate_choice_model_results


def _softmax(values: np.ndarray) -> np.ndarray:
    scaled = values - values.max()
    exp_values = np.exp(scaled)
    return exp_values / exp_values.sum()


def _build_choice_table(coefficients: dict[str, float], rng: np.random.Generator, *, n_sets: int) -> pd.DataFrame:
    titles = [f"Title {i}" for i in range(8)]
    rows = []

    for experiment_number in range(1, n_sets + 1):
        titles_perm = rng.permutation(titles)
        utilities = []
        alternatives = []

        for position, title in enumerate(titles_perm):
            price = float(rng.lognormal(mean=3.0, sigma=0.4))
            rating_count = int(rng.poisson(lam=50))
            rating = float(np.clip(rng.normal(loc=4.1, scale=0.35), 1.0, 5.0))
            sponsored = bool(rng.random() < 0.2)
            overall_pick = bool(rng.random() < 0.1)
            low_stock = bool(rng.random() < 0.15)

            row = position // 4 + 1
            column = position % 4 + 1
            utility = (
                coefficients["row_1_dummy"] * (1 if row == 1 else 0)
                + coefficients["col_1_dummy"] * (1 if column == 1 else 0)
                + coefficients["col_2_dummy"] * (1 if column == 2 else 0)
                + coefficients["col_3_dummy"] * (1 if column == 3 else 0)
                + coefficients["log_price"] * np.log(price)
                + coefficients["log_rating_count"] * np.log(rating_count + 1)
                + coefficients["rating"] * rating
                + coefficients["sponsored_tag"] * int(sponsored)
                + coefficients["overall_pick_tag"] * int(overall_pick)
                + coefficients["scarcity_tag"] * int(low_stock)
            )

            alternatives.append(
                {
                    "query": "q1",
                    "experiment_label": "exp1",
                    "experiment_number": experiment_number,
                    "model_name": "m1",
                    "title": title,
                    "assigned_position": position,
                    "price": price,
                    "rating_count": rating_count,
                    "rating": rating,
                    "sponsored": sponsored,
                    "overall_pick": overall_pick,
                    "low_stock": low_stock,
                }
            )
            utilities.append(utility)

        probabilities = _softmax(np.array(utilities))
        chosen_idx = rng.choice(len(alternatives), p=probabilities)
        for idx, alt in enumerate(alternatives):
            alt["selected"] = 1 if idx == chosen_idx else 0
            rows.append(alt)

    df = pd.DataFrame(rows)

    # Ensure each title is selected at least once so the filter does not drop it.
    selection_counts = df.groupby("title")["selected"].sum()
    for title in selection_counts[selection_counts == 0].index:
        row_idx = df[df["title"] == title].index[0]
        experiment_number = df.loc[row_idx, "experiment_number"]
        df.loc[df["experiment_number"] == experiment_number, "selected"] = 0
        df.loc[row_idx, "selected"] = 1

    return df


def test_choice_model_recovers_coefficients(tmp_path):
    rng = np.random.default_rng(7)
    true_coefficients = {
        "row_1_dummy": 0.2,
        "col_1_dummy": 0.1,
        "col_2_dummy": -0.05,
        "col_3_dummy": 0.05,
        "log_price": -0.3,
        "log_rating_count": 0.12,
        "rating": 0.35,
        "sponsored_tag": -0.08,
        "overall_pick_tag": 0.25,
        "scarcity_tag": 0.05,
    }

    input_path = tmp_path / "choice_model.csv"
    output_path = tmp_path / "choice_model_results.csv"
    table = _build_choice_table(true_coefficients, rng, n_sets=500)
    table.to_csv(input_path, index=False)

    generate_choice_model_results(input_path, output_path)
    results = pd.read_csv(output_path, index_col=0)
    estimated = results["m1"]

    for name, expected in true_coefficients.items():
        assert np.isclose(estimated[name], expected, rtol=0.3, atol=0.2)
