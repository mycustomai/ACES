import asyncio
import atexit
import os
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd
from rich import print as _print

from agent.src.environment import start_environment
from agent.src.exceptions import AgentException
from agent.src.logger import create_logger
from agent.src.shopper import SimulatedShopper
from agent.src.typedefs import EngineParams
from agent.src.types import TargetSite
from experiments.config import ExperimentData
from experiments.data_loader import experiments_iter, load_experiment_data
from experiments.results import (aggregate_model_data, aggregate_run_data,
                                 continue_experiment_or_exit)
from experiments.server import start_fastapi_server, stop_fastapi_server
from sandbox import set_experiment_data


class BaseEvaluationRuntime(ABC):
    """
    Base class for evaluation runtimes that handles dataset loading and output directory creation.
    """

    def __init__(
        self,
        dataset_name: str,
        output_dir_override: Optional[str] = None,
        debug_mode: bool = False,
    ):
        """
        Initialize the BaseEvaluationRuntime.

        Args:
            dataset_name: Name for the dataset (used for output directory)
            output_dir_override: Optional override for output directory name
            debug_mode: Show full tracebacks and skip try/except handling
        """
        self.dataset_name = dataset_name
        self.output_dir_override = output_dir_override
        self.debug_mode = debug_mode

        # Set up output directory
        self._setup_output_dir()

    def _setup_output_dir(self):
        """Set up the output directory based on dataset name."""
        if self.output_dir_override:
            self.run_output_dir = Path(f"experiment_logs/{self.output_dir_override}")
        else:
            self.run_output_dir = Path(f"experiment_logs/{self.dataset_name}")

        # Ensure output directory exists
        os.makedirs(self.run_output_dir, exist_ok=True)

    @property
    def experiments_iter(self) -> Iterable[ExperimentData]:
        """Abstract property that returns an iterator over experiments."""
        raise NotImplementedError


class SimpleEvaluationRuntime(BaseEvaluationRuntime):
    """
    Consolidated runtime for running AI agent evaluation experiments.

    This class consolidates the functionality from run.py and experiments/runner.py
    into a single, simplified interface for running experiments across multiple models.
    """

    def __init__(
        self,
        local_dataset_path: str,
        engine_params_list: List[EngineParams],
        output_dir_override: Optional[str] = None,
        max_concurrent_models: int = 10,
        experiment_count_limit: Optional[int] = None,
        experiment_label_filter: Optional[str] = None,
        debug_mode: bool = False,
    ):
        """
        Initialize the SimpleEvaluationRuntime.

        Args:
            local_dataset_path: Path to the dataset CSV file
            engine_params_list: List of model engine parameters to evaluate
            output_dir_override: Optional override for output directory name
            max_concurrent_models: Maximum concurrent models per experiment
            experiment_count_limit: Number of experiments to run (None = no limit)
            experiment_label_filter: Filter experiments by specific label (None = no filter)
            debug_mode: Show full tracebacks and skip try/except handling
        """
        # Extract dataset name from path
        dataset_filename = os.path.splitext(os.path.basename(local_dataset_path))[0]
        super().__init__(dataset_filename, output_dir_override, debug_mode)

        # Load the dataset
        self.local_dataset_path = local_dataset_path
        self.dataset = load_experiment_data(local_dataset_path)

        self.engine_params_list = engine_params_list
        self.max_concurrent_models = max_concurrent_models
        self.experiment_count_limit = experiment_count_limit
        self.experiment_label_filter = experiment_label_filter
        self.server = None

        # Register cleanup function
        atexit.register(self._cleanup)

    @property
    def experiments_iter(self) -> Iterable[ExperimentData]:
        """Return an iterator over experiments from the local dataset."""
        return experiments_iter(self.dataset, self.dataset_name)

    def _start_server(self) -> bool:
        """Start the FastAPI server."""
        _print("[bold blue]Starting FastAPI server...")
        self.server = start_fastapi_server()
        if not self.server:
            _print("[bold red]Failed to start server.")
            return False
        _print("[bold green]FastAPI server started on http://127.0.0.1:5000")
        return True

    def _cleanup(self):
        """Cleanup function to stop server."""
        if self.server:
            _print("[bold blue]Stopping FastAPI server...")
            stop_fastapi_server()
            _print("[bold green]FastAPI server stopped.")

    def run_single_experiment(self, data: ExperimentData, engine_params: EngineParams):
        """
        Run a single experiment with the given data and engine parameters.

        Args:
            data: ExperimentData containing experiment configuration
            engine_params: Engine parameters for the model
        """
        experiment_df = data.experiment_df.copy()

        # Check if experiment already exists
        if data.experiment_label and data.experiment_number is not None:
            journey_dir = data.journey_dir(self.run_output_dir, engine_params)
            experiment_csv_path = journey_dir / "experiment_data.csv"

            if experiment_csv_path.exists():
                _print(
                    f"[purple]Skipping experiment - experiment_data.csv already exists: {experiment_csv_path}"
                )
                return

        model_output_dir = data.model_output_dir(self.run_output_dir, engine_params)

        with start_environment(
            TargetSite.MOCKAMAZON, product_name=data.query
        ) as environment:
            with create_logger(
                data.query,
                output_dir=model_output_dir,
                experiment_df=experiment_df,
                engine_params=engine_params,
                experiment_label=data.experiment_label,
                experiment_number=data.experiment_number,
            ) as logger:
                shopper = SimulatedShopper(
                    initial_message=data.prompt_template,
                    engine_params=engine_params,
                    environment=environment,
                    logger=logger,
                )

                if self.debug_mode:
                    _print(
                        "[bold yellow]DEBUG_MODE enabled - full tracebacks will be shown"
                    )
                    shopper.run()
                    _print("[bold green]Experiment completed successfully.")
                else:
                    try:
                        shopper.run()
                    except KeyboardInterrupt:
                        continue_experiment_or_exit()
                    except AgentException as e:
                        _print(f"[bold orange]Agent exception: {e}")
                    except Exception as e:
                        _print(f"[bold red]Error: {e}")
                        warnings.warn(f"Unknown error: {e}")
                    else:
                        _print("[bold green]Experiment completed successfully.")

        aggregate_model_data(model_output_dir)

    async def run_single_experiment_parallel_models(self, data: ExperimentData):
        """
        Run a single experiment across multiple models in parallel.

        Args:
            data: ExperimentData containing experiment configuration
        """
        _print(
            f"\n[underline yellow]STARTING EXPERIMENT - {data.query} - {data.experiment_label} - Exp # {data.experiment_number} - {len(self.engine_params_list)} models\n"
        )

        # Set experiment data in the sandbox (shared across all models for this experiment)
        set_experiment_data(data.experiment_df)

        semaphore = asyncio.Semaphore(self.max_concurrent_models)

        async def run_model_experiment(engine_params):
            async with semaphore:
                _print(
                    f"[bold cyan]Running {engine_params.config_name} for {data.query} - {data.experiment_label} - Exp #{data.experiment_number}"
                )

                await asyncio.to_thread(self.run_single_experiment, data, engine_params)

                _print(
                    f"[bold green]Completed {engine_params.config_name} for {data.query} - {data.experiment_label} - Exp #{data.experiment_number}"
                )

        # Run all models for this experiment in parallel
        model_tasks = [
            run_model_experiment(engine_params)
            for engine_params in self.engine_params_list
        ]
        await asyncio.gather(*model_tasks)

    async def run_all_experiments(self):
        """
        Run all experiments with multiple models in parallel per experiment.
        """
        current_experiment = 1

        for data in experiments_iter(self.dataset, self.dataset_name):
            if (
                self.experiment_count_limit is not None
                and current_experiment > self.experiment_count_limit
            ):
                _print(
                    f"[bold yellow]Experiment count limit reached. Stopping at {self.experiment_count_limit} experiments."
                )
                break

            if (
                self.experiment_label_filter is not None
                and data.experiment_label != self.experiment_label_filter
            ):
                _print(
                    f"[bold yellow]Skipping experiment {data.experiment_label} (filtering for {self.experiment_label_filter})"
                )
                continue

            await self.run_single_experiment_parallel_models(data)
            current_experiment += 1

    async def run(self):
        """
        Main entry point to run all experiments.

        This method:
        1. Starts the FastAPI server
        2. Runs all experiments across all models
        3. Aggregates the results
        4. Cleans up resources
        """
        # Start the FastAPI server
        if not self._start_server():
            raise RuntimeError("Failed to start FastAPI server")

        _print(f"[bold blue]Created run directory: {self.run_output_dir}")
        _print(
            f"[bold green]Loaded {len(self.engine_params_list)} model configurations"
        )

        try:
            # Run experiments with parallel model execution
            await self.run_all_experiments()

            # Aggregate data across all models in this run
            _print(f"\n[bold blue]Aggregating run data across all models...")
            aggregate_run_data(str(self.run_output_dir))

        finally:
            self._cleanup()
