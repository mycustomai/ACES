"""
Worker service for managing concurrent experiment execution.

Centralizes worker management patterns used by screenshot runtimes.
"""

import asyncio
import os
import time
from pathlib import Path
from typing import Any, Callable, Iterable, List, Optional

from rich import print as _print
from rich.progress import Progress

from agent.src.environment import BaseShoppingEnvironment
from agent.src.typedefs import EngineParams
from experiments.config import ExperimentData


class ExperimentWorkerService:
    """Service for managing concurrent experiment execution."""

    def __init__(self, max_concurrent_per_engine: int = 5):
        self.max_concurrent_per_engine = max_concurrent_per_engine

    async def run_experiments(
        self,
        experiments: Iterable[ExperimentData],
        engines: List[EngineParams],
        create_environment_fn: Callable[[ExperimentData], BaseShoppingEnvironment],
        run_single_experiment_fn: Callable[[ExperimentData, EngineParams], Any],
        distributed_engines: Optional[List[EngineParams]] = None,
        experiment_count_limit: Optional[int] = None,
        experiment_label_filter: Optional[str] = None,
    ) -> None:
        """
        Run experiments concurrently across multiple engines with proper load balancing.

        Args:
            experiments: Iterable of experiment data
            engines: List of engine parameters (original engines for organization)
            distributed_engines: List of distributed engine parameters with multiple API keys for load balancing
            create_environment_fn: Function to create shopping environment
            run_single_experiment_fn: Function to run a single experiment
            experiment_count_limit: Maximum number of experiments to run
            experiment_label_filter: Filter experiments by label
        """
        # Use distributed engines if provided, otherwise fall back to original engines
        if distributed_engines is None:
            distributed_engines = engines
        # Group experiments by engine
        experiments_per_engine = {}

        # Group distributed engine params by original engine type
        engine_param_groups: dict[str, list[EngineParams]] = {}
        for engine_params in engines:
            engine_key = f"{engine_params.engine_type}_{engine_params.model}"
            experiments_per_engine[engine_key] = []
            engine_param_groups[engine_key] = [
                ep
                for ep in distributed_engines
                if f"{ep.engine_type}_{ep.model}" == engine_key
            ]

        # Collect and filter experiments
        current_experiment = 1
        for data in experiments:
            if experiment_count_limit and current_experiment > experiment_count_limit:
                break

            if (
                experiment_label_filter
                and data.experiment_label != experiment_label_filter
            ):
                continue

            # Add to all engines
            for engine_params in engines:
                engine_key = f"{engine_params.engine_type}_{engine_params.model}"
                experiments_per_engine[engine_key].append(data)

            current_experiment += 1

        # Calculate total tasks
        total_tasks = sum(
            len(experiments_per_engine[f"{ep.engine_type}_{ep.model}"])
            for ep in engines
        )

        if total_tasks == 0:
            _print("[bold yellow]No experiments to run")
            return

        _print(
            f"[bold blue]Starting {total_tasks} experiments across {len(engines)} engines..."
        )

        # Print experiments per provider
        _print("[bold cyan]Experiments per provider:")
        for engine_params in engines:
            engine_key = f"{engine_params.engine_type}_{engine_params.model}"
            experiment_count = len(experiments_per_engine[engine_key])
            if experiment_count > 0:
                _print(f"  {engine_params.config_name}: {experiment_count} experiments")

        # Run experiments with progress tracking
        with Progress() as progress:
            task_id = progress.add_task("Running experiments", total=total_tasks)
            completed_count = 0
            start_time = time.time()

            def update_progress():
                nonlocal completed_count
                completed_count += 1
                elapsed_time = time.time() - start_time
                rate = completed_count / elapsed_time if elapsed_time > 0 else 0
                progress.update(
                    task_id,
                    completed=completed_count,
                    description=f"Running experiments ({rate:.1f}/s, {elapsed_time:.1f}s)",
                )

            # Create workers for each engine
            workers = []
            queues = {}

            for engine_params in engines:
                engine_key = f"{engine_params.engine_type}_{engine_params.model}"
                experiments = experiments_per_engine[engine_key]
                distributed_params = engine_param_groups[engine_key]

                if not experiments:
                    continue

                _print(
                    f"[bold blue]{engine_params.config_name}: {len(experiments)} experiments"
                )
                _print(
                    f"[bold blue]  Using {len(distributed_params)} workers (max {self.max_concurrent_per_engine} concurrent per worker)"
                )

                # Create queue for this engine
                queue = asyncio.Queue()
                queues[engine_key] = queue

                # Create workers for this engine (max_concurrent_per_engine workers per engine)
                for i in range(
                    min(len(distributed_params), self.max_concurrent_per_engine)
                ):
                    distributed_param = distributed_params[i % len(distributed_params)]
                    worker = asyncio.create_task(
                        self._experiment_worker(
                            queue,
                            distributed_param,
                            run_single_experiment_fn,
                            update_progress,
                        )
                    )
                    workers.append(worker)

                # Add experiments to queue
                for data in experiments:
                    await queue.put(data)

            # Wait for all experiments to complete
            for queue in queues.values():
                await queue.join()

            # Stop all workers
            for queue in queues.values():
                for _ in range(self.max_concurrent_per_engine):
                    await queue.put(None)  # Sentinel to stop workers

            # Wait for workers to finish
            await asyncio.gather(*workers, return_exceptions=True)

            end_time = time.time()

            _print(
                f"[bold cyan]Experiment completion summary ({end_time - start_time:.1f}s):"
            )
            _print(f"[bold green]  ✓ Completed: {completed_count}")

    async def _experiment_worker(
        self,
        queue: asyncio.Queue,
        engine_params: EngineParams,
        run_single_experiment_fn: Callable,
        progress_callback=None,
    ):
        """Worker that processes experiments from a queue one at a time."""
        # DO NOT set gRPC environment variables as they cause deadlocks with Gemini models

        while True:
            try:
                data = await queue.get()
                if data is None:  # Sentinel value to stop worker
                    break

                await run_single_experiment_fn(data, engine_params)

                if progress_callback:
                    progress_callback()

            except Exception as e:
                # Check if this is an authentication/API key error that should be re-raised
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
                        f"[bold red]Authentication error for {engine_params.config_name}: {e}"
                    )
                    raise  # Re-raise authentication errors to stop execution
                else:
                    _print(
                        f"[bold red]Failed experiment for {engine_params.config_name}: {e}"
                    )
                    if progress_callback:
                        progress_callback()
            finally:
                queue.task_done()
