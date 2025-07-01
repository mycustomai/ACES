"""
HuggingFace Hub dataset screenshot runtime implementation.

This module provides the HFHubDatasetRuntime class for running experiments
with screenshots embedded in HuggingFace Hub datasets.
"""

from typing import Iterable, List, Optional

from rich import print as _print

from agent.src.environment import BaseShoppingEnvironment
from agent.src.typedefs import EngineParams
from experiments.config import ExperimentData
from experiments.data_loader import hf_experiments_iter
from experiments.dataset_environment import DatasetShoppingEnvironment

from .base import BaseScreenshotRuntime


class HFHubDatasetRuntime(BaseScreenshotRuntime):
    """
    Screenshot-based runtime for HuggingFace Hub datasets.

    This runtime loads datasets from HuggingFace Hub where each row contains
    a complete experiment with screenshots stored in the dataset.
    """

    def __init__(
        self,
        engine_params_list: List[EngineParams],
        hf_dataset_name: str,
        subset: str = "all",
        output_dir_override: Optional[str] = None,
        max_concurrent_per_engine: int = 5,
        experiment_count_limit: Optional[int] = None,
        experiment_label_filter: Optional[str] = None,
        debug_mode: bool = False,
    ):
        """
        Initialize the HFHubDatasetRuntime.

        Args:
            engine_params_list: List of model engine parameters to evaluate
            hf_dataset_name: Name of the dataset on HuggingFace Hub
            subset: Subset/configuration of the dataset to load
            output_dir_override: Optional override for output directory name
            max_concurrent_per_engine: Maximum concurrent experiments per engine type
            experiment_count_limit: Number of experiments to run (None = no limit)
            experiment_label_filter: Filter experiments by specific label (None = no filter)
            debug_mode: Show full tracebacks and skip try/except handling
        """
        # Use HF dataset name and subset as the dataset name
        dataset_name = f"{hf_dataset_name.replace('/', '_')}_{subset}"

        # Call parent with dataset name
        super().__init__(
            dataset_name=dataset_name,
            engine_params_list=engine_params_list,
            output_dir_override=output_dir_override,
            max_concurrent_per_engine=max_concurrent_per_engine,
            experiment_count_limit=experiment_count_limit,
            experiment_label_filter=experiment_label_filter,
            debug_mode=debug_mode,
        )

        self.hf_dataset_name = hf_dataset_name
        self.subset = subset

        _print(
            f"[bold blue]Using {hf_dataset_name} (subset: {subset}) from HuggingFace Hub..."
        )

    @property
    def experiments_iter(self) -> Iterable[ExperimentData]:
        """Return an iterator over experiments from the HF dataset."""
        return hf_experiments_iter(self.hf_dataset_name, subset=self.subset)

    def get_experiments_dataframe(self):
        """HF datasets don't have a dataframe representation."""
        return None

    def get_dataset_path(self) -> Optional[str]:
        """HF datasets don't have a local path."""
        return None

    def validate_prerequisites(self) -> bool:
        """HF datasets don't need screenshot validation."""
        return True

    def create_shopping_environment(
        self, data: ExperimentData
    ) -> BaseShoppingEnvironment:
        """Create a dataset shopping environment with the screenshot from the dataset."""
        # The screenshot should be attached to the ExperimentData by hf_experiments_iter
        screenshot = data.screenshot

        if screenshot is None:
            raise ValueError(
                f"No screenshot found for experiment ({data.query}, {data.experiment_label}, {data.experiment_number})"
            )

        return DatasetShoppingEnvironment(screenshot_image=screenshot)
