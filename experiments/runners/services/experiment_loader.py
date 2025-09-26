from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from agent.src.typedefs import EngineConfigName, EngineParams
from experiments.config import ExperimentId
from experiments.data_loader import (ExperimentData, experiments_iter,
                                     hf_experiments_iter)
from experiments.runners.batch_runtime.common.encoded_id_mixin import (
    EncodedExperimentIdMixin,
)
from experiments.utils.dataset_ops import get_dataset_name


class _DatasetSource(ABC):
    """Internal protocol for dataset loading strategies."""

    @abstractmethod
    def get_dataset_name(self) -> str:
        """Return the dataset name used for directory/file naming."""

    @abstractmethod
    def load_experiments(
        self, experiment_count_limit: Optional[int] = None
    ) -> list[ExperimentData]:
        """Load experiments from this dataset source."""

    @abstractmethod
    def get_screenshots_dir(self) -> Optional[Path]:
        """Return the screenshots directory if one exists."""

    @abstractmethod
    def get_dataset_path(self) -> Optional[str]:
        """Return the local dataset path when available."""

class _LocalDatasetSource(_DatasetSource):
    """Local CSV dataset with filesystem screenshots."""

    def __init__(self, local_dataset_path: str):
        self.local_dataset_path = local_dataset_path
        self._dataset_name = get_dataset_name(local_dataset_path)
        self._screenshots_dir = (
            Path(local_dataset_path).parent / 'screenshots' / self._dataset_name
        )

    def get_dataset_name(self) -> str:
        return self._dataset_name

    def load_experiments(
        self, experiment_count_limit: Optional[int] = None
    ) -> list[ExperimentData]:
        dataframe = pd.read_csv(self.local_dataset_path)
        experiments = list(experiments_iter(dataframe, self._dataset_name))
        if experiment_count_limit is not None:
            return experiments[:experiment_count_limit]
        return experiments

    def get_screenshots_dir(self) -> Optional[Path]:
        return self._screenshots_dir

    def get_dataset_path(self) -> Optional[str]:
        return self.local_dataset_path

class _HFDatasetSource(_DatasetSource):
    """HuggingFace Hub dataset with embedded screenshots."""

    def __init__(self, hf_dataset_name: str, subset: str = 'all'):
        self.hf_dataset_name = hf_dataset_name
        self.subset = subset
        self._dataset_name = f"{hf_dataset_name.replace('/', '_')}_{subset}"

    def get_dataset_name(self) -> str:
        return self._dataset_name

    def load_experiments(
        self, experiment_count_limit: Optional[int] = None
    ) -> list[ExperimentData]:
        experiments = list(
            hf_experiments_iter(self.hf_dataset_name, subset=self.subset)
        )
        for experiment in experiments:
            experiment.dataset_name = self._dataset_name
        if experiment_count_limit is not None:
            return experiments[:experiment_count_limit]
        return experiments

    def get_screenshots_dir(self) -> Optional[Path]:
        return None

    def get_dataset_path(self) -> Optional[str]:
        return None

class ExperimentLoader(EncodedExperimentIdMixin):
    """Universal experiment loader supporting local and HuggingFace datasets."""

    def __init__(
        self,
        engine_params: list[EngineParams],
        experiment_count_limit: Optional[int] = None,
        local_dataset_path: Optional[str] = None,
        hf_dataset_name: Optional[str] = None,
        hf_subset: str = 'all',
    ):
        if local_dataset_path and hf_dataset_name:
            raise ValueError(
                'Cannot specify both local_dataset_path and hf_dataset_name'
            )
        if not local_dataset_path and not hf_dataset_name:
            raise ValueError(
                'Must specify either local_dataset_path or hf_dataset_name'
            )

        if local_dataset_path:
            self._source: _DatasetSource = _LocalDatasetSource(local_dataset_path)
        else:
            assert hf_dataset_name is not None  # Narrow type for mypy
            self._source = _HFDatasetSource(hf_dataset_name, subset=hf_subset)

        self.engine_params = engine_params
        self.experiment_count_limit = experiment_count_limit
        self.experiments = set(
            self._source.load_experiments(experiment_count_limit)
        )

    @property
    def dataset_name(self) -> str:
        return self._source.get_dataset_name()

    @property
    def screenshots_dir(self) -> Optional[Path]:
        return self._source.get_screenshots_dir()

    @property
    def dataset_path(self) -> Optional[str]:
        return self._source.get_dataset_path()

    def experiments_iter(self) -> Iterable[ExperimentData]:
        return iter(self.experiments)

    def load_outstanding_experiments(
        self, submitted_experiments: dict[EngineConfigName, list[ExperimentId]]
    ) -> dict[EngineConfigName, list[ExperimentData]]:
        """Load experiments that have not yet been submitted."""
        outstanding_experiments: dict[EngineConfigName, list[ExperimentData]] = {}
        for engine in self.engine_params:
            outstanding_experiments[engine.config_name] = []
            existing = set(submitted_experiments.get(engine.config_name, []))
            outstanding_experiments[engine.config_name] = list(
                self.experiments.difference(existing)
            )
        return outstanding_experiments

    def get_experiment_by_id(self, experiment_id: ExperimentId) -> ExperimentData:
        """Retrieve an experiment by its ID, supporting encoded IDs."""
        experiment_ids = [exp.experiment_id for exp in self.experiments]
        resolved_id = self.resolve_experiment_id(experiment_id, experiment_ids)
        try:
            return next(
                experiment
                for experiment in self.experiments
                if experiment.experiment_id == resolved_id
            )
        except StopIteration as exc:
            raise KeyError(f"No experiment found with id {experiment_id!r}") from exc
