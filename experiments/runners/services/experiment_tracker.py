import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from agent.src.typedefs import EngineConfigName, EngineParams
from experiments.config import ExperimentId
from experiments.runners.batch_new.common.encoded_id_mixin import \
    EncodedExperimentIdMixin
from experiments.runners.batch_new.typedefs import (BatchStatus,
                                                    ExperimentFailureModes,
                                                    ExperimentResult,
                                                    ExperimentSubmissionRecord,
                                                    ProviderBatchId)


class ExperimentTracker:
    """Manages persistence of completed, submitted, and queued experiment submissions.

    Attributes:
        completed:
            Mapping of engine config names to completed experiment records.
        in_progress:
            Mapping of engine config names to in-progress experiment records.
        failed:
            Mapping of engine config names to failed experiment records.

            Note, failed experiments are considered submitted and therefore not retried.
    """

    def __init__(self, engine_params_list: list[EngineParams], run_output_dir: Path):
        self.engine_params_list = engine_params_list
        self.run_output_dir = run_output_dir
        self.batch_metadata_dir = run_output_dir / "batch_metadata"

        # Create directory structure
        self.batch_metadata_dir.mkdir(exist_ok=True)
        for engine in engine_params_list:
            provider_dir = self.batch_metadata_dir / engine.config_name
            provider_dir.mkdir(exist_ok=True)
            for status in [
                BatchStatus.COMPLETED,
                BatchStatus.IN_PROGRESS,
                BatchStatus.FAILED,
            ]:
                (provider_dir / status).mkdir(exist_ok=True)

        self.completed: dict[EngineConfigName, list[ExperimentSubmissionRecord]] = {
            engine.config_name: [] for engine in engine_params_list
        }
        self.in_progress: dict[EngineConfigName, list[ExperimentSubmissionRecord]] = {
            engine.config_name: [] for engine in engine_params_list
        }
        self.failed: dict[EngineConfigName, list[ExperimentSubmissionRecord]] = {
            engine.config_name: [] for engine in engine_params_list
        }

    @property
    def has_batches_in_progress(self) -> bool:
        return any(self.in_progress.values())

    @property
    def batch_ids_in_progress(self) -> dict[EngineConfigName, list[ProviderBatchId]]:
        """Return a mapping of batch provider ids which are currently in progress.

        Used for monitoring batch completion.
        """
        return {
            config_name: list(set([job.batch_id for job in jobs]))
            for config_name, jobs in self.in_progress.items()
        }

    @property
    def in_progress_batch_count(self) -> int:
        """Get the total number of batches across all providers that are in progress.

        Returns:
            Total number of batches currently in progress.
        """
        return sum(len(batch_ids) for batch_ids in self.batch_ids_in_progress.values())

    def _get_completed_dir(self, config_name: EngineConfigName) -> Path:
        return self.batch_metadata_dir / config_name / BatchStatus.COMPLETED

    def _get_in_progress_dir(self, config_name: EngineConfigName) -> Path:
        return self.batch_metadata_dir / config_name / BatchStatus.IN_PROGRESS

    def _get_failed_dir(self, config_name: EngineConfigName) -> Path:
        return self.batch_metadata_dir / config_name / BatchStatus.FAILED

    @staticmethod
    def _load_experiment_dir(target: Path) -> Iterable[ExperimentSubmissionRecord]:
        if not target.exists():
            raise ValueError(f"Directory does not exist: {target}")
        for json_file in target.glob("*.json"):
            with open(json_file, "r") as f:
                job_data = json.load(f)
                yield ExperimentSubmissionRecord(**job_data)

    def load_submitted_experiments(self) -> dict[EngineConfigName, list[ExperimentId]]:
        """Load existing experiments from the 'batch_metadata' directory and return experiment IDs of those which have already been submitted."""
        submitted_experiments = {}

        for engine in self.engine_params_list:
            config_name = engine.config_name
            submitted_experiments[config_name] = []

            dir_container_pairs = [
                (self._get_failed_dir, self.failed),
                (self._get_in_progress_dir, self.in_progress),
                (self._get_completed_dir, self.completed),
            ]
            for dir_getter, container in dir_container_pairs:
                directory = dir_getter(config_name)
                for job in self._load_experiment_dir(directory):
                    container[config_name].append(job)
                    submitted_experiments[config_name].append(job.experiment_id)

        return submitted_experiments

    def get_experiment_ids_for_batch(
        self, batch_id: ProviderBatchId
    ) -> list[ExperimentId]:
        """Get experiment IDs associated with a batch ID.

        Args:
            batch_id: The provider batch ID to look up.

        Returns:
            List of experiment IDs associated with the batch.
        """
        experiment_ids = []
        for engine_config, records in self.in_progress.items():
            for submission_record in records:
                if submission_record.batch_id == batch_id:
                    experiment_ids.append(submission_record.experiment_id)
        return experiment_ids

    def set_experiments_in_progress(
        self, submitted: dict[EngineConfigName, list[ExperimentSubmissionRecord]]
    ) -> None:
        """Save multiple records as 'in progress' and persist to disk.

        Args:
            submitted: Dictionary mapping engine config names to lists of submitted records
                      to mark as in-progress.
        """
        for config_name, records in submitted.items():
            self.in_progress[config_name].extend(records)

            for job in records:
                job_file = (
                    self._get_in_progress_dir(config_name) / f"{job.experiment_id}.json"
                )
                with open(job_file, "w") as f:
                    job_dict = job.model_dump()
                    # Exclude failure_reason if it's None
                    if job_dict.get("failure_reason") is None:
                        job_dict.pop("failure_reason", None)
                    json.dump(job_dict, f, indent=2)

    def set_experiment_failed(self, record: ExperimentSubmissionRecord) -> None:
        config_name = record.config_name
        self.failed[config_name].append(record)
        self.in_progress[config_name].remove(record)

        in_progress_file = (
            self._get_in_progress_dir(config_name) / f"{record.experiment_id}.json"
        )
        failed_file = self._get_failed_dir(config_name) / f"{record.experiment_id}.json"

        if in_progress_file.exists():
            in_progress_file.rename(failed_file)
        else:
            raise RuntimeError(f"In-progress file does not exist: {in_progress_file}")

    def set_experiment_complete(
        self, batch_id: ProviderBatchId, config_name: EngineConfigName
    ) -> None:
        """Move submission records from in-progress to completed based on batch_id.

        Args:
            batch_id: The provider batch ID to complete.
            config_name: The engine config name for the batch.

        Raises:
            ValueError: If no in-progress records found with the given batch_id.
            RuntimeError: If the in-progress file does not exist on disk.
        """
        # find all associated experiment records
        associated_experiments: list[ExperimentSubmissionRecord] = []
        for engine_config, records in self.in_progress.items():
            for submission_record in records:
                if submission_record.batch_id == batch_id:
                    associated_experiments.append(submission_record.model_copy())

        if not associated_experiments:
            raise ValueError(
                f"No in-progress experiments found with batch_id: {batch_id}"
            )

        for exp in associated_experiments:
            self.in_progress[config_name].remove(exp)
        self.completed[config_name].extend(associated_experiments)

        # move the associated file from in_progress to completed
        for submission_record in associated_experiments:
            in_progress_file = (
                self._get_in_progress_dir(config_name)
                / f"{submission_record.experiment_id}.json"
            )
            completed_file = (
                self._get_completed_dir(config_name)
                / f"{submission_record.experiment_id}.json"
            )

            if in_progress_file.exists():
                in_progress_file.rename(completed_file)
            else:
                raise RuntimeError(
                    f"In-progress file does not exist: {in_progress_file}"
                )

    def filter_failed_results(
        self, data: list[ExperimentResult]
    ) -> tuple[list[ExperimentResult], dict[ExperimentFailureModes, int]]:
        """Filter out failed experiment results and move corresponding submission records to failed state.

        Args:
            data: List of experiment results to filter

        Returns:
            List of non-failed experiment results
        """
        failed_results = [result for result in data if result.failure]
        successful_results = [result for result in data if result.success]
        failure_reasons = defaultdict(int)

        # Process failed results
        for failed_result in failed_results:
            submission_record = None
            all_experiment_ids = [
                record.experiment_id
                for records in self.in_progress.values()
                for record in records
            ]
            failed_experiment_id = EncodedExperimentIdMixin.resolve_experiment_id(
                failed_result.experiment_id, all_experiment_ids
            )

            for config_name, records in self.in_progress.items():
                for record in records:
                    if record.experiment_id == failed_experiment_id:
                        submission_record = record
                        break
                if submission_record:
                    break

            if submission_record:
                reason = failed_result.failure_reason
                submission_record.failure_result = failed_result.model_copy(deep=True)
                failure_reasons[reason] += 1
                self.set_experiment_failed(submission_record)
            else:
                raise ValueError(
                    f"No in-progress submission record found for experiment_id: {failed_result.experiment_id}"
                )

        return successful_results, failure_reasons
