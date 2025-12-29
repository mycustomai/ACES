from enum import StrEnum
from pathlib import Path

import pandas as pd
import yaml
from experiments.engine_loader import load_all_model_engine_params


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
    """Load model display names from EngineParams YAML config files.

    Returns:
        Dict[model_name, human-friendly name]
    """
    engine_params = load_all_model_engine_params()
    display_name_mapping = {
        engine.model: engine.display_name
        for engine in engine_params
    }
    return display_name_mapping


def get_rationality_suite_experiment_names() -> dict[str, dict[str, str]]:
    path = REPO_ROOT / "config" / "experiment_metadata" / "rationality_suite_experiment_names.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def filter_valid_experiments(orig_df: pd.DataFrame) -> pd.DataFrame:
    """ Remove invalid experiments from input df. Input df is mutated.

    See Also:
        `agent/src/logger.py` for how the model outputs are parsed and products
        are selected.
    """
    df = orig_df.copy()

    df["selected"] = df["selected"].clip(lower=0, upper=1)

    valid_experiments = df.groupby(["model_name", "experiment_label", "query", "experiment_number"]).filter(
        lambda x: x["selected"].sum() == 1
    )
    return valid_experiments

