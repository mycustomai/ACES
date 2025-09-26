import json
import time
from copy import copy
from itertools import chain
from pathlib import Path
from time import sleep
from typing import List, Optional

from rich import print as _print
from rich.progress import BarColumn, Progress, SpinnerColumn, TimeElapsedColumn

from agent.src.core.tools import AddToCartTool
from agent.src.typedefs import EngineConfigName, EngineParams, EngineType
from experiments.config import ExperimentData
from experiments.results import aggregate_run_data
from experiments.runners.batch_runtime.providers import (anthropic, base, gemini,
                                                         openai)
from experiments.runners.batch_runtime.typedefs import (BatchRequest, BatchStatus,
                                                        BatchStatusResult,
                                                        ExperimentSubmissionRecord,
                                                        SerializedBatchRequest)
from experiments.runners.services import (ExperimentLoader, ExperimentTracker,
                                          GCSManager,
                                          ScreenshotValidationService)
from experiments.runners.services.agent_simulator import AgentSimulator
from experiments.runners.simple_runtime import BaseEvaluationRuntime

DEFAULT_MONITOR_INTERVAL = 10


class BatchOrchestratorRuntime(BaseEvaluationRuntime):
    def __init__(
        self,
        engine_params_list: List[EngineParams],
        remote: bool,
        output_dir_override: Optional[str] = None,
        experiment_count_limit: Optional[int] = None,
        experiment_label_filter: Optional[str] = None,
        debug_mode: bool = False,
        force_submit: bool = False,
        monitor_interval: int = DEFAULT_MONITOR_INTERVAL,
        local_dataset_path: Optional[str] = None,
        hf_dataset_name: Optional[str] = None,
        hf_subset: Optional[str] = None,
    ):
        """Initialize the simplified BatchEvaluationRuntime."""
        self.experiment_loader = ExperimentLoader(
            engine_params=engine_params_list,
            experiment_count_limit=experiment_count_limit,
            local_dataset_path=local_dataset_path,
            hf_dataset_name=hf_dataset_name,
            hf_subset=hf_subset,
        )

        dataset_name = self.experiment_loader.dataset_name

        super().__init__(dataset_name, output_dir_override, debug_mode)

        self._hf_materialized_dir: Optional[Path] = None
        self.screenshots_dir = self.experiment_loader.screenshots_dir
        if self.screenshots_dir is None:
            self.screenshots_dir = self._materialize_hf_screenshots()

        self.experiment_tracker = ExperimentTracker(
            engine_params_list=engine_params_list,
            run_output_dir=self.run_output_dir,
        )

        dataset_path = self.experiment_loader.dataset_path
        self.dataset_path = dataset_path
        self.remote = remote
        self.hf_dataset_name = hf_dataset_name
        self.hf_subset = hf_subset

        self.screenshot_validator: Optional[ScreenshotValidationService] = None
        if self.screenshots_dir is not None:
            self.screenshot_validator = ScreenshotValidationService(
                screenshots_dir=self.screenshots_dir,
                dataset_name=self.dataset_name,
                debug_mode=self.debug_mode,
            )

        self.screenshot_manager: Optional[GCSManager] = None
        if self.remote:
            if self.screenshots_dir is None:
                self.screenshots_dir = self._materialize_hf_screenshots()
            self.screenshot_manager = GCSManager(
                dataset_name=self.dataset_name,
                screenshots_dir=self.screenshots_dir,
            )
        self.simulator = AgentSimulator(
            dataset_name=self.dataset_name,
            run_output_dir=self.run_output_dir,
            use_remote=self.remote,
            local_dataset_path=dataset_path,
            hf_dataset_name=hf_dataset_name,
            hf_subset=hf_subset,
            screenshots_dir=self.screenshots_dir,
            verbose=self.debug_mode,
        )

        self.engine_params_list = engine_params_list
        self.monitor_interval = monitor_interval
        self.experiment_count_limit = experiment_count_limit

        self._setup_provider_tools()

    def _materialize_hf_screenshots(self) -> Path:
        """Persist screenshots from a HuggingFace dataset to the filesystem."""
        if self._hf_materialized_dir is not None:
            return self._hf_materialized_dir

        base_dir = self.run_output_dir / "materialized_screenshots"
        dataset_dir = base_dir / self.dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)

        for experiment in self.experiment_loader.experiments:
            screenshot = getattr(experiment, "screenshot", None)
            if screenshot is None:
                raise ValueError(
                    "HuggingFace dataset missing embedded screenshot for experiment"
                )

            target_path = experiment.get_local_screenshot_path(base_dir)
            target_path.parent.mkdir(parents=True, exist_ok=True)

            if target_path.exists():
                continue

            if isinstance(screenshot, bytes):
                target_path.write_bytes(screenshot)
            elif hasattr(screenshot, "save"):
                screenshot.save(target_path, format="PNG")
            elif hasattr(screenshot, "to_pil"):
                screenshot.to_pil().save(target_path, format="PNG")
            elif isinstance(screenshot, dict) and "bytes" in screenshot:
                target_path.write_bytes(screenshot["bytes"])
            else:
                raise TypeError(
                    "Unsupported screenshot payload type for HuggingFace dataset"
                )

        self._hf_materialized_dir = base_dir
        return base_dir

    def _setup_provider_tools(self) -> None:
        self.provider_deserializers: dict[EngineConfigName, base.Deserializer] = {}
        self.provider_monitors: dict[EngineConfigName, base.Monitor] = {}
        self.provider_serializers: dict[EngineConfigName, base.Serializer] = {}
        self.provider_submitters: dict[EngineConfigName, base.Submitter] = {}

        for engine in self.engine_params_list:
            engine_type: EngineType = engine.engine_type
            config_name = engine.config_name

            # noinspection PyUnreachableCode
            match engine_type:
                case EngineType.ANTHROPIC:
                    self.provider_submitters[config_name] = anthropic.Submitter(engine)
                    self.provider_serializers[config_name] = anthropic.Serializer(
                        engine
                    )
                    self.provider_monitors[config_name] = anthropic.Monitor(engine)
                    self.provider_deserializers[config_name] = anthropic.Deserializer(
                        engine
                    )
                case EngineType.OPENAI:
                    self.provider_submitters[config_name] = openai.Submitter(engine)
                    self.provider_serializers[config_name] = openai.Serializer(engine)
                    self.provider_monitors[config_name] = openai.Monitor(engine)
                    self.provider_deserializers[config_name] = openai.Deserializer(
                        engine
                    )
                case EngineType.GEMINI:
                    self.provider_submitters[config_name] = gemini.Submitter(engine)
                    self.provider_serializers[config_name] = gemini.Serializer(engine)
                    self.provider_monitors[config_name] = gemini.Monitor(engine)
                    self.provider_deserializers[config_name] = gemini.Deserializer(
                        engine
                    )
                case _:
                    raise ValueError(f"Unsupported engine type: {engine_type}")

    def _load_experiments(self) -> dict[EngineConfigName, list[ExperimentData]]:
        """Select, load, and prepare environment."""
        submitted_experiments = self.experiment_tracker.load_submitted_experiments()
        outstanding_experiments = self.experiment_loader.load_outstanding_experiments(
            submitted_experiments
        )

        if any(outstanding_experiments.values()):
            unique_experiments = list(
                set(chain.from_iterable(outstanding_experiments.values()))
            )
            if self.screenshot_validator:
                self.screenshot_validator.validate_all_screenshots(
                    unique_experiments,
                    self.dataset_path,
                )
            if self.screenshot_manager:
                self.screenshot_manager.upload(
                    outstanding_experiments.values(), verbose=self.debug_mode
                )

        return outstanding_experiments

    def _save_serialized_batch_requests(
        self,
        provider_batch_requests: dict[EngineConfigName, list[SerializedBatchRequest]],
    ) -> None:
        """Save serialized batch requests to batch_metadata/{config_name} directory when in debug mode."""
        for config_name, serialized_requests in provider_batch_requests.items():
            batch_metadata_dir = (
                self.run_output_dir / "batch_metadata" / config_name / "batch_requests"
            )
            batch_metadata_dir.mkdir(parents=True, exist_ok=True)

            # Save all serialized requests in JSONL format
            filepath = batch_metadata_dir / "batch_requests.jsonl"

            with open(filepath, "w") as f:
                for serialized_request in serialized_requests:
                    # Convert serialized request to JSON-serializable format
                    request_data = serialized_request.provider_request
                    json.dump(request_data, f, default=str)
                    f.write("\n")

            _print(
                f"[dim]Saved {len(serialized_requests)} serialized batch requests to {filepath}"
            )

    def run(self):
        outstanding_experiments = self._load_experiments()

        # prepare submissions
        batch_requests: dict[EngineConfigName, BatchRequest] = {}
        for engine_params in self.engine_params_list:
            experiments = copy(outstanding_experiments[engine_params.config_name])
            if not experiments:
                _print(f"[dim]No experiments for [blue]{engine_params.config_name}")
                continue
            raw_messages = [
                self.simulator.create_experiment_request(experiment, engine_params)
                for experiment in experiments
            ]
            request = BatchRequest(
                experiments=experiments,
                raw_messages=raw_messages,
                engine_params=engine_params,
                tools=[AddToCartTool()],
            )
            batch_requests[engine_params.config_name] = request

        # note: serialization performs chunking of `BatchRequest`
        provider_batch_requests: dict[
            EngineConfigName, list[SerializedBatchRequest]
        ] = {}
        for config_name, request in batch_requests.items():
            serialized = self.provider_serializers[config_name].serialize(request)
            provider_batch_requests[config_name] = serialized

        if self.debug_mode:
            self._save_serialized_batch_requests(provider_batch_requests)

        # submit
        submission_records: dict[
            EngineConfigName, list[ExperimentSubmissionRecord]
        ] = {}
        for config_name, request in provider_batch_requests.items():
            records = self.provider_submitters[config_name].submit(request)
            submission_records[config_name] = records
        self.experiment_tracker.set_experiments_in_progress(submission_records)

        self._monitor_batches()

        _print("[green]Aggregating run data... ")
        aggregate_run_data(str(self.run_output_dir))
        _print("[bold green]Batch processing complete!")

    def _monitor_batches(self) -> None:
        """Monitor submitted batches until completion."""
        _print("[bold blue]Starting batch monitoring...")

        with Progress(
            SpinnerColumn(),
            "[blue]{task.fields[config_name]}",
            "[progress.description]{task.description}",
            BarColumn(bar_width=80),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeElapsedColumn(),
            console=None,
            transient=True,
        ) as progress:
            monitor_task = progress.add_task(
                "Monitoring batches...",
                total=self.experiment_tracker.in_progress_batch_count,
                config_name="",
            )

            engine_params_lookup = {
                params.config_name: params for params in self.engine_params_list
            }

            while self.experiment_tracker.has_batches_in_progress:
                for (
                    config_name,
                    batch_ids,
                ) in self.experiment_tracker.batch_ids_in_progress.items():
                    engine_params = engine_params_lookup[config_name]

                    def progress_callback(*args, **kwargs):
                        progress.update(
                            monitor_task, *args, **kwargs, config_name=config_name
                        )

                    progress_callback(
                        description=f"Fetching {len(batch_ids)} batches..."
                    )

                    batch_results = self.provider_monitors[config_name].monitor_batches(
                        batch_ids
                    )

                    for batch in batch_results:
                        self._handle_monitor_result(
                            batch, engine_params, progress_callback
                        )

                progress.update(monitor_task, description="Sleeping...", config_name="")

                sleep(self.monitor_interval)

    def _handle_monitor_result(
        self,
        batch: BatchStatusResult,
        engine_params: EngineParams,
        progress_callback: Optional[Progress.update] = None,
    ) -> None:
        experiment_ids = self.experiment_tracker.get_experiment_ids_for_batch(
            batch.batch_id
        )

        config_name = engine_params.config_name

        # noinspection PyUnreachableCode
        match batch.status:
            case BatchStatus.COMPLETED:
                if progress_callback:
                    progress_callback(description="Deserializing")
                _deserialized_results = self.provider_deserializers[
                    config_name
                ].deserialize(batch.result)

                # ensure deserialized results are complete
                result_experiment_ids = [
                    result.experiment_id for result in _deserialized_results.data
                ]

                assert len(result_experiment_ids) == len(experiment_ids), (
                    f"Not all ids processed. Expected {len(experiment_ids)}, got {len(result_experiment_ids)}"
                )

                # process successful results
                if progress_callback:
                    progress_callback(description="Processing Results")

                successful_results, failure_reasons = (
                    self.experiment_tracker.filter_failed_results(
                        _deserialized_results.data
                    )
                )

                if not successful_results:
                    _print(
                        f"[bold red]Failed to process any results for batch {batch.batch_id}."
                    )
                    _print(
                        f"[bold red]Retrieved {len(result_experiment_ids)} results. All failed."
                    )
                    _print(f"[bold red]Reason: {failure_reasons}")
                    return

                matching_experiment_data: list[ExperimentData] = [
                    self.experiment_loader.get_experiment_by_id(r.experiment_id)
                    for r in successful_results
                ]

                # Timer for processing step
                processing_start_time = time.time()
                for result, experiment in zip(
                    successful_results, matching_experiment_data, strict=True
                ):
                    processed = self.simulator.process_experiment_result(
                        experiment, engine_params, result
                    )
                    if not processed:
                        _print(
                            f"[bold red]Failed to process experiment: {experiment.experiment_id}"
                        )
                processing_end_time = time.time()

                processing_duration = processing_end_time - processing_start_time
                if self.debug_mode and successful_results:
                    avg_time_per_experiment = processing_duration / len(
                        successful_results
                    )
                    _print(
                        f"[dim]Processing completed in {processing_duration:.2f}s (avg: {avg_time_per_experiment:.2f}s per experiment)"
                    )

                self.experiment_tracker.set_experiment_complete(
                    batch.batch_id, config_name
                )

                if self.debug_mode:
                    _print(
                        f"[dim]Finished processing {len(result_experiment_ids)} results"
                    )
                    for reason, count in failure_reasons.items():
                        _print(f"[dim red]Failed {count} results: {reason}")

                if progress_callback:
                    progress_callback(advance=1)
            case BatchStatus.FAILED:
                # TODO: handle failed statuses
                raise NotImplementedError("Got failed batch status.")
            case BatchStatus.IN_PROGRESS:
                return
            case _:
                raise ValueError(f"Unexpected batch status: {batch.status}")
