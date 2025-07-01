from typing import Optional

import pandas as pd

from agent.src.typedefs import EngineConfigName, EngineParams
from experiments.config import ExperimentId
from experiments.data_loader import ExperimentData, experiments_iter
from experiments.runners.batch_new.common.encoded_id_mixin import \
    EncodedExperimentIdMixin
from experiments.utils.dataset_ops import get_dataset_name


class ExperimentLoader(EncodedExperimentIdMixin):
    """Manages experiments by loading experiments from local or HF dataset."""

    def __init__(
        self,
        local_dataset_path: str,
        engine_params: list[EngineParams],
        experiment_count_limit: Optional[int] = None,
    ):
        # TODO: handle hf dataset
        self.local_dataset_path = local_dataset_path
        self.engine_params = engine_params
        self.experiment_count_limit = experiment_count_limit

        self.experiments = set(self._load_local_dataset())

    def _load_local_dataset(self) -> list[ExperimentData]:
        df = pd.read_csv(self.local_dataset_path)
        dataset_name = get_dataset_name(self.local_dataset_path)
        data = list(experiments_iter(df, dataset_name))
        if self.experiment_count_limit:
            return data[: self.experiment_count_limit]
        return data

    def load_outstanding_experiments(
        self, submitted_experiments: dict[EngineConfigName, list[ExperimentId]]
    ) -> dict[EngineConfigName, list[ExperimentData]]:
        """Load experiments that are not yet submitted."""
        outstanding_experiments = {}
        for engine in self.engine_params:
            outstanding_experiments[engine.config_name] = []
            existing = set(submitted_experiments.get(engine.config_name, []))

            outstanding_experiments[engine.config_name] = list(
                self.experiments.difference(existing)
            )

        return outstanding_experiments

    def get_experiment_by_id(self, experiment_id: ExperimentId) -> ExperimentData:
        """Get an experiment by its ID."""
        experiment_ids = [exp.experiment_id for exp in self.experiments]
        resolved_id = self.resolve_experiment_id(experiment_id, experiment_ids)

        try:
            return next(
                exp for exp in self.experiments if exp.experiment_id == resolved_id
            )
        except StopIteration:
            raise KeyError(f"No experiment found with id {experiment_id!r}")
