"""
Base class for screenshot-based runtimes using composed services.

This module provides the abstract base class for all screenshot-based experiment runtimes
that use the new service architecture for better code reuse and maintainability.
"""

import os
from abc import ABC, abstractmethod
from typing import Iterable, List, Optional

from rich import print as _print

from agent.src.environment import BaseShoppingEnvironment
from agent.src.exceptions import AgentException
from agent.src.logger import create_logger
from agent.src.shopper import SimulatedShopper
from agent.src.typedefs import EngineParams, EngineType
from experiments.config import ExperimentData
from experiments.results import aggregate_run_data
from experiments.runners.services import (ExperimentWorkerService,
                                          ScreenshotValidationService)
from experiments.runners.simple_runtime import BaseEvaluationRuntime


class BaseScreenshotRuntime(BaseEvaluationRuntime, ABC):
    """
    Simplified base class for screenshot-based runtimes using composed services.

    This class uses the new service architecture to reduce code duplication
    and improve maintainability across different screenshot runtime implementations.
    """

    def __init__(
        self,
        dataset_name: str,
        engine_params_list: List[EngineParams],
        output_dir_override: Optional[str] = None,
        max_concurrent_per_engine: int = 5,
        experiment_count_limit: Optional[int] = None,
        experiment_label_filter: Optional[str] = None,
        debug_mode: bool = False,
    ):
        """
        Initialize the BaseScreenshotRuntime with services.

        Args:
            dataset_name: Name for the dataset (used for output directory)
            engine_params_list: List of model engine parameters to evaluate
            output_dir_override: Optional override for output directory name
            max_concurrent_per_engine: Maximum concurrent experiments per engine type
            experiment_count_limit: Number of experiments to run (None = no limit)
            experiment_label_filter: Filter experiments by specific label (None = no filter)
            debug_mode: Show full tracebacks and skip try/except handling
        """
        super().__init__(dataset_name, output_dir_override, debug_mode)

        self.engine_params_list = engine_params_list
        self.experiment_count_limit = experiment_count_limit
        self.experiment_label_filter = experiment_label_filter
        self.max_concurrent_per_engine = max_concurrent_per_engine

        self.distributed_engine_params = self._distribute_engine_params()

        self.worker_service = ExperimentWorkerService(
            max_concurrent_per_engine=max_concurrent_per_engine
        )

        # Screenshot validation service will be initialized by subclasses
        # TODO: validation service for HFHub runner should upload to GCS when `remote` enabled
        self.validation_service: Optional[ScreenshotValidationService] = None

        _print(
            f"[bold blue]Configured {len(engine_params_list)} engines with {max_concurrent_per_engine} max concurrent per engine"
        )

    @property
    @abstractmethod
    def experiments_iter(self) -> Iterable[ExperimentData]:
        """Abstract property that returns an iterator over experiments."""
        pass

    @abstractmethod
    def create_shopping_environment(
        self, data: ExperimentData
    ) -> BaseShoppingEnvironment:
        """Create a shopping environment for the given experiment data."""
        pass

    def validate_prerequisites(self) -> bool:
        """Validate any prerequisites before running experiments."""
        if self.validation_service:
            return self.validation_service.validate_all_screenshots(
                self.get_experiments_dataframe(), self.get_dataset_path()
            )
        return True

    @abstractmethod
    def get_experiments_dataframe(self):
        """Get the experiments dataframe for validation."""
        pass

    @abstractmethod
    def get_dataset_path(self) -> Optional[str]:
        """Get the dataset path for screenshot regeneration."""
        pass

    async def run_single_experiment(
        self, data: ExperimentData, engine_params: EngineParams
    ):
        """
        Run a single experiment with the given data and engine parameters using screenshots.

        Args:
            data: ExperimentData containing experiment configuration
            engine_params: Engine parameters for the model
        """
        if self.debug_mode:
            _print(
                f"[bold blue]Running {engine_params.config_name} - {data.query}_{data.experiment_label}_{data.experiment_number}"
            )

        # Check if experiment already exists
        if data.experiment_label and data.experiment_number is not None:
            journey_dir = data.journey_dir(self.run_output_dir, engine_params)
            experiment_csv_path = journey_dir / "experiment_data.csv"

            if experiment_csv_path.exists():
                return

        model_output_dir = data.model_output_dir(self.run_output_dir, engine_params)

        # Create shopping environment
        environment = self.create_shopping_environment(data)

        try:
            experiment_df = data.experiment_df.copy()

            with create_logger(
                data.query,
                output_dir=model_output_dir,
                experiment_df=experiment_df,
                engine_params=engine_params,
                experiment_label=data.experiment_label,
                experiment_number=data.experiment_number,
                silent=True,
            ) as logger:
                shopper = SimulatedShopper(
                    initial_message=data.prompt_template,
                    engine_params=engine_params,
                    environment=environment,
                    logger=logger,
                )

                if self.debug_mode:
                    await shopper.arun()
                else:
                    try:
                        await shopper.arun()
                    except AgentException as e:
                        _print(
                            f"[bold orange]Agent exception in {engine_params.config_name}: {e}"
                        )
                        return
                    except Exception as e:
                        # Check if this is a key error that should be re-raised
                        error_msg = str(e).lower()
                        if any(
                            keyword in error_msg
                            for keyword in [
                                "authentication",
                                "auth",
                                "api key",
                                "api_key",
                                "unauthorized",
                                "invalid key",
                                "permission denied",
                                "forbidden",
                                "401",
                                "403",
                            ]
                        ):
                            _print(
                                f"[bold red]Authentication error in {engine_params.config_name}: {e}"
                            )
                            raise
                        else:
                            _print(
                                f"[bold red]Error in {engine_params.config_name}: {e}"
                            )
                            return

        except Exception as e:
            _print(
                f"[bold red]Failed {engine_params.config_name} - {data.query}_{data.experiment_label}_{data.experiment_number}: {e}"
            )

    async def run(self):
        """
        Main entry point to run all experiments using services.

        This method:
        1. Validates prerequisites using screenshot validation service
        2. Runs experiments using worker service
        3. Aggregates the results
        """
        if not self.validate_prerequisites():
            raise RuntimeError(
                "Prerequisites validation failed. Cannot proceed with experiments."
            )

        _print(f"[bold blue]Created run directory: {self.run_output_dir}")
        _print(
            f"[bold green]Loaded {len(self.engine_params_list)} engine configurations"
        )

        # Use worker service to run experiments with distributed engine params for load balancing
        await self.worker_service.run_experiments(
            experiments=self.experiments_iter,
            engines=self.engine_params_list,
            create_environment_fn=self.create_shopping_environment,
            run_single_experiment_fn=self.run_single_experiment,
            # TODO: remove all distributed engine functionality. See https://github.com/mycustomai/ACES/issues/9
            #distributed_engines=self.distributed_engine_params,
            experiment_count_limit=self.experiment_count_limit,
            experiment_label_filter=self.experiment_label_filter,
        )

        # Aggregate data across all models in this run
        _print(f"\n[purple]Aggregating run data across all engines...")
        aggregate_run_data(str(self.run_output_dir))

        _print(f"[bold green]All experiments completed successfully!")

    def _distribute_engine_params(self) -> List[EngineParams]:
        """
        Create distributed engine parameters with multiple API keys for load balancing.

        This method creates multiple copies of each engine parameter configuration,
        each using a different API key to distribute load across multiple keys.
        """
        distributed_params = []

        for engine_params in self.engine_params_list:
            # Load all available API keys for this engine type
            api_keys = self.load_api_keys_for_provider(engine_params.engine_type)

            if not api_keys:
                # If no API keys found, use the original params
                distributed_params.append(engine_params)
                continue

            # Create copies with different API keys
            for i, api_key in enumerate(api_keys):
                params_copy = EngineParams(
                    engine_type=engine_params.engine_type,
                    model=engine_params.model,
                    api_key=api_key,
                    temperature=engine_params.temperature,
                    max_new_tokens=engine_params.max_new_tokens,
                )

                # Copy any additional parameters
                for attr_name in dir(engine_params):
                    if not attr_name.startswith("_") and hasattr(
                        params_copy, attr_name
                    ):
                        if attr_name not in [
                            "engine_type",
                            "model",
                            "config_name",
                            "api_key",
                            "temperature",
                            "max_new_tokens",
                        ]:
                            try:
                                setattr(
                                    params_copy,
                                    attr_name,
                                    getattr(engine_params, attr_name),
                                )
                            except AttributeError:
                                pass  # Skip read-only attributes

                distributed_params.append(params_copy)

        _print(
            f"[bold blue]Created {len(distributed_params)} distributed engine configurations from {len(self.engine_params_list)} original engines"
        )
        return distributed_params

    @staticmethod
    def load_api_keys_for_provider(engine_type: EngineType) -> List[str]:
        """
        Load multiple API keys from environment variables for a given provider.

        Environment variable patterns:
        - OPENAI_API_KEY, OPENAI_API_KEY_2, OPENAI_API_KEY_3, etc.
        - ANTHROPIC_API_KEY, ANTHROPIC_API_KEY_2, ANTHROPIC_API_KEY_3, etc.
        - GOOGLE_API_KEY, GOOGLE_API_KEY_2, GOOGLE_API_KEY_3, etc.

        Args:
            engine_type: The engine type (openai, anthropic, gemini, etc.)

        Returns:
            List of API keys found for the provider
        """
        api_keys = []

        base_env_var = engine_type.env_var_prefix
        if not base_env_var:
            _print(
                f"[bold yellow]Warning: Unknown engine type {engine_type}, no API keys loaded"
            )
            return api_keys

        primary_key = os.getenv(base_env_var)
        if primary_key:
            api_keys.append(primary_key)

        # Load additional API keys (numbered 2, 3, 4, etc.)
        key_index = 2
        while True:
            additional_key = os.getenv(f"{base_env_var}_{key_index}")
            if additional_key:
                api_keys.append(additional_key)
                key_index += 1
            else:
                break

        if api_keys:
            _print(f"[bold green]Loaded {len(api_keys)} API key(s) for {engine_type}")
        else:
            _print(f"[bold yellow]Warning: No API keys found for {engine_type}")

        return api_keys
