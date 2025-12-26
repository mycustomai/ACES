from enum import StrEnum
from pathlib import Path

import pandas as pd
import yaml


class SanityCheckMode(StrEnum):
    PRICE = "price"
    RATING = "rating"
    INSTRUCTION = "instruction"
    AR_PRICE = "ar_price"


# TODO: fix unsatisfactory pattern. Define global location for `config` dir
REPO_ROOT = Path(__file__).parent.parent.parent


def load_query_shortnames() -> dict[str, dict[str, str]]:
    """Load product shortnames from YAML config."""
    path = REPO_ROOT / "config" / "experiment_metadata" / "product_shortnames.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def load_model_display_names() -> dict[str, str]:
    """Load model display names from YAML config."""
    path = REPO_ROOT / "config" / "experiment_metadata" / "model_display_names.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def get_rationality_suite_experiment_names() -> dict[str, dict[str, str]]:
    path = REPO_ROOT / "config" / "experiment_metadata" / "rationality_suite_experiment_names.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def filter_valid_experiments(df: pd.DataFrame) -> pd.DataFrame:
    """ Select experiments in which a product was selected.

    Note:
        `1` denotes that the model returned the correct, exact product name.
        `2` denotes that the product was selected based on the title only.
        Any negative sum indicates an unrecoverable error, and as such, the
        experiment must be ignored.

    See Also:
        `agent/src/logger.py` for how the model outputs are parsed and products
        are selected.
    """
    valid_experiments = df.groupby(["model_name", "experiment_label", "query", "experiment_number"]).filter(
        lambda x: x["selected"].sum() >= 1
    )
    return valid_experiments

