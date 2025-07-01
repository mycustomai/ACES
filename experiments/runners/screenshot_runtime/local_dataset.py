"""
Local dataset screenshot runtime implementation.

This module provides the LocalDatasetRuntime class for running experiments
with pre-captured screenshots stored in the filesystem.
"""

from pathlib import Path
from typing import Iterable, List, Optional

from rich import print as _print

from agent.src.environment import BaseShoppingEnvironment
from agent.src.typedefs import EngineParams
from experiments.config import ExperimentData
from experiments.data_loader import experiments_iter, load_experiment_data
from experiments.filesystem_environment import FilesystemShoppingEnvironment
from experiments.runners.services.screenshot_validation import \
    ScreenshotValidationService

from .base import BaseScreenshotRuntime


class LocalDatasetRuntime(BaseScreenshotRuntime):
    """
    Screenshot-based runtime for local datasets with pre-captured screenshots.

    This runtime reads screenshots from the filesystem that were previously
    captured and stored alongside the dataset CSV file.
    """

    def __init__(
        self,
        local_dataset_path: str,
        engine_params_list: List[EngineParams],
        output_dir_override: Optional[str] = None,
        max_concurrent_per_engine: int = 5,
        experiment_count_limit: Optional[int] = None,
        experiment_label_filter: Optional[str] = None,
        debug_mode: bool = False,
        remote: bool = False,
    ):
        """
        Initialize the LocalDatasetRuntime.

        Args:
            local_dataset_path: Path to the dataset CSV file (required)
            engine_params_list: List of model engine parameters to evaluate
            output_dir_override: Optional override for output directory name
            max_concurrent_per_engine: Maximum concurrent experiments per engine type
            experiment_count_limit: Number of experiments to run (None = no limit)
            experiment_label_filter: Filter experiments by specific label (None = no filter)
            debug_mode: Show full tracebacks and skip try/except handling
            remote: Use remote GCS URLs instead of local screenshot bytes
        """
        # Extract dataset name from path
        dataset_name = Path(local_dataset_path).stem.replace("_dataset", "")

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

        self.local_dataset_path = local_dataset_path
        self.remote = remote

        # Load the dataset
        self.dataset = load_experiment_data(local_dataset_path)

        # Set up screenshots directory
        dataset_dir = Path(local_dataset_path).parent
        self.screenshots_dir = dataset_dir / "screenshots"
        self.dataset_name = dataset_name

        # Initialize screenshot validation service
        self.validation_service = ScreenshotValidationService(
            self.screenshots_dir, str(dataset_dir)
        )

        _print(f"[bold blue]Screenshot directory: {self.screenshots_dir}")

    @property
    def experiments_iter(self) -> Iterable[ExperimentData]:
        """Return an iterator over experiments from the local dataset."""
        return experiments_iter(self.dataset, self.dataset_name)

    def get_experiments_dataframe(self):
        """Get the experiments dataframe for validation."""
        return self.dataset

    def get_dataset_path(self) -> Optional[str]:
        """Get the dataset path for screenshot regeneration."""
        return self.local_dataset_path

    def create_shopping_environment(
        self, data: ExperimentData
    ) -> BaseShoppingEnvironment:
        """Create a filesystem shopping environment."""
        return FilesystemShoppingEnvironment(
            screenshots_dir=self.screenshots_dir,
            query=data.query,
            experiment_label=data.experiment_label,
            experiment_number=data.experiment_number,
            dataset_name=self.dataset_name,
            remote=self.remote,
        )
