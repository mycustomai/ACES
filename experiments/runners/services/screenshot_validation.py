"""
Screenshot validation service for batch and screenshot runtimes.
"""

from pathlib import Path

from rich import print as _print
from rich.progress import (BarColumn, Progress, SpinnerColumn, TextColumn,
                           TimeRemainingColumn)

from experiments.config import ExperimentData
from experiments.utils.screenshot_collector import (
    collect_screenshots_parallel, ensure_all_screenshots_exist,
    get_missing_screenshots)


class ScreenshotValidationService:
    """Unified screenshot validation across all runtimes."""

    def __init__(
        self, screenshots_dir: Path, dataset_name: str, debug_mode: bool = False
    ):
        self.screenshots_dir = Path(screenshots_dir)
        self.dataset_name = dataset_name
        self.debug_mode = debug_mode

    def validate_all_screenshots(
        self, experiments: list[ExperimentData], local_dataset_path: str
    ) -> bool:
        """
        Validate that all required screenshots exist.

        Returns:
            True if all screenshots exist and are valid
        """
        _print("[bold blue]Validating screenshot availability...")

        if not ensure_all_screenshots_exist(experiments, local_dataset_path):
            _print(
                "[bold yellow]Some screenshots are missing. Attempting to regenerate..."
            )

            # Try to regenerate missing screenshots
            success = self._generate_screenshots(experiments, local_dataset_path)
            if not success:
                raise RuntimeError("Failed to regenerate screenshots")

        _print("[bold green]All required screenshots are available and valid.")
        return True

    def _generate_screenshots(
        self,
        experiments: list[ExperimentData],
        dataset_path: str,
        parallel: bool = True,
        num_workers: int = 24,
    ) -> bool:
        """
        Generate missing screenshots.

        Returns:
            True if generation was successful
        """
        try:
            if parallel:
                _print(
                    f"[bold blue]Generating missing screenshots with {num_workers} workers..."
                )

                missing_experiments = get_missing_screenshots(experiments, dataset_path)

                # Create progress bar
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                    TextColumn("({task.completed}/{task.total})"),
                    TimeRemainingColumn(),
                    console=None,
                ) as progress:
                    task = progress.add_task(
                        "Generating screenshots...", total=len(missing_experiments)
                    )

                    def progress_callback(completed: int):
                        progress.update(task, completed=completed)

                    collect_screenshots_parallel(
                        missing_experiments,
                        dataset_path,
                        num_workers,
                        verbose=self.debug_mode,
                        progress_callback=progress_callback,
                    )
            else:
                raise NotImplementedError("Sequential generation no longer supported")

            # Ensure all screenshots were created
            _print("[bold blue]Validating generation...")
            if ensure_all_screenshots_exist(experiments, dataset_path):
                _print("[bold green]All screenshots successfully generated.")
                return True
            else:
                _print(
                    "[bold red]Screenshot regeneration failed. Some screenshots are still missing."
                )
                return False

        except Exception as e:
            _print(f"[bold red]Error during screenshot regeneration: {e}")
            return False
