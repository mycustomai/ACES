"""
GCS upload service for batch and screenshot runtimes.

Handles uploading screenshots to Google Cloud Storage while maintaining
the same directory structure as local storage.
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from rich import print as _print
from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn

from experiments.data_loader import experiments_iter

try:
    from google.cloud import storage

    GCS_AVAILABLE = True
except ImportError:
    GCS_AVAILABLE = False
    _print("[bold yellow]Warning: Google Cloud Storage not available")
    _print("[bold yellow]Install with: pip install google-cloud-storage")


class GCSUploadService:
    """Service for uploading screenshots to Google Cloud Storage."""

    def __init__(
        self, bucket_name: Optional[str] = None, project_id: Optional[str] = None
    ):
        """
        Initialize GCS upload service.

        Args:
            bucket_name: GCS bucket name (defaults to GCS_BUCKET_NAME env var)
            project_id: Google Cloud project ID (defaults to GOOGLE_CLOUD_PROJECT env var)
        """
        self.bucket_name = bucket_name or os.getenv("GCS_BUCKET_NAME")
        self.project_id = project_id or os.getenv("GOOGLE_CLOUD_PROJECT")

        if not self.bucket_name:
            raise ValueError(
                "GCS bucket name must be provided or set in GCS_BUCKET_NAME environment variable"
            )

        self._storage_client = None
        self._bucket = None

        # Validate GCS availability
        if not GCS_AVAILABLE:
            raise ImportError(
                "Google Cloud Storage library not available. Install with: pip install google-cloud-storage"
            )

    def _ensure_storage_client(self) -> bool:
        """Ensure GCS storage client is initialized."""
        if self._storage_client is None:
            try:
                self._storage_client = storage.Client(project=self.project_id)
                self._bucket = self._storage_client.bucket(self.bucket_name)
                _print(
                    f"[bold green]✓ GCS client initialized for bucket: {self.bucket_name}"
                )
                return True
            except Exception as e:
                _print(f"[bold red]Failed to initialize GCS client: {e}")
                return False
        return True

    def upload_screenshot(self, local_path: Path, gcs_path: str) -> bool:
        """
        Upload a single screenshot to GCS.

        Args:
            local_path: Local path to the screenshot file
            gcs_path: GCS path (without gs:// prefix)

        Returns:
            True if upload was successful
        """
        if not self._ensure_storage_client():
            return False

        if not local_path.exists():
            return False

        try:
            blob = self._bucket.blob(gcs_path)
            blob.upload_from_filename(str(local_path))
            return True
        except Exception as e:
            # More concise error logging to avoid spam
            return False

    def upload_screenshots_batch(
        self,
        screenshots_dir: Path,
        dataset_name: str,
        experiments_df: pd.DataFrame,
        max_workers: int = 6,
        skip_existing: bool = True,
    ) -> Dict[str, Any]:
        """
        Upload screenshots for experiments to GCS with progress tracking.

        Args:
            screenshots_dir: Local directory containing screenshots
            dataset_name: Name of the dataset for GCS path structure
            experiments_df: DataFrame containing experiment information
            max_workers: Number of concurrent upload threads
            skip_existing: Skip files that already exist in GCS (default: True)

        Returns:
            Dictionary with upload results and statistics
        """
        if not self._ensure_storage_client():
            return {"success": False, "error": "Failed to initialize GCS client"}

        # Get all screenshot paths that need to be uploaded
        upload_tasks = self._prepare_upload_tasks(
            screenshots_dir, dataset_name, experiments_df, skip_existing
        )

        if not upload_tasks:
            _print("[bold yellow]No screenshots found to upload")
            return {"success": True, "uploaded": 0, "failed": 0, "skipped": 0}

        # Track upload results
        results = {
            "success": True,
            "uploaded": 0,
            "failed": 0,
            "skipped": 0,
            "failed_files": [],
        }

        # Create progress bar
        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("({task.completed}/{task.total})"),
            TimeRemainingColumn(),
            console=None,
        ) as progress:
            task_id = progress.add_task(
                "Uploading screenshots", total=len(upload_tasks)
            )

            # Upload files concurrently
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all upload tasks
                future_to_task = {
                    executor.submit(self._upload_single_screenshot, task): task
                    for task in upload_tasks
                }

                # Process completed uploads
                for future in as_completed(future_to_task):
                    task = future_to_task[future]
                    try:
                        success = future.result()
                        if success:
                            results["uploaded"] += 1
                        else:
                            results["failed"] += 1
                            results["failed_files"].append(str(task["local_path"]))
                    except Exception as e:
                        results["failed"] += 1
                        results["failed_files"].append(str(task["local_path"]))
                        _print(f"[bold red]Exception during upload: {e}")

                    progress.advance(task_id)

        # Print summary
        if results["failed"] > 0:
            results["success"] = False

        # Always show completion
        total_processed = results["uploaded"] + results["failed"]
        if total_processed > 0:
            _print(
                f"[bold green]✓ Upload complete: {results['uploaded']}/{total_processed} successful"
            )

        return results

    def _prepare_upload_tasks(
        self,
        screenshots_dir: Path,
        dataset_name: str,
        experiments_df: pd.DataFrame,
        skip_existing: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Prepare list of upload tasks based on experiments.

        Args:
            screenshots_dir: Local screenshots directory
            dataset_name: Dataset name for GCS path structure
            experiments_df: DataFrame with experiment data
            skip_existing: Skip files that already exist in GCS

        Returns:
            List of upload task dictionaries
        """
        potential_tasks = []
        all_experiments = list(experiments_iter(experiments_df, dataset_name))
        total_experiments = len(all_experiments)
        processed_count = 0

        try:
            # First pass: collect all potential tasks with local files that exist
            with Progress(
                TextColumn("[bold blue]Checking local files"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TextColumn("({task.completed}/{task.total})"),
                console=None,
            ) as progress:
                task_id = progress.add_task(
                    "Checking experiments", total=total_experiments
                )

                for data in all_experiments:
                    processed_count += 1
                    progress.advance(task_id)

                    # Use ExperimentData methods for standardized path construction
                    local_screenshot_path = data.get_local_screenshot_path(
                        screenshots_dir
                    )

                    # Skip if local file doesn't exist
                    if not local_screenshot_path.exists():
                        continue

                    # Use ExperimentData method for GCS path
                    gcs_path = data.get_gcs_screenshot_path()

                    potential_tasks.append(
                        {
                            "local_path": local_screenshot_path,
                            "gcs_path": gcs_path,
                            "query": data.query,
                            "experiment_label": data.experiment_label,
                            "experiment_number": data.experiment_number,
                        }
                    )

            # Second pass: batch check which files already exist in GCS
            upload_tasks = potential_tasks
            if skip_existing and potential_tasks:
                gcs_paths = [task["gcs_path"] for task in potential_tasks]
                existence_map = self.batch_check_screenshots_exist(gcs_paths)

                # Filter out existing files
                upload_tasks = [
                    task
                    for task in potential_tasks
                    if not existence_map.get(task["gcs_path"], False)
                ]

        except Exception as e:
            _print(f"[bold red]Error preparing upload tasks: {e}")
            raise

        return upload_tasks

    def _upload_single_screenshot(self, task: Dict[str, Any]) -> bool:
        """
        Upload a single screenshot task.

        Args:
            task: Upload task dictionary with local_path and gcs_path

        Returns:
            True if upload was successful
        """
        return self.upload_screenshot(task["local_path"], task["gcs_path"])

    def check_screenshot_exists_in_gcs(self, gcs_path: str) -> bool:
        """
        Check if a screenshot already exists in GCS.

        Args:
            gcs_path: GCS path (without gs:// prefix)

        Returns:
            True if the file exists in GCS
        """
        if not self._ensure_storage_client():
            return False

        try:
            blob = self._bucket.blob(gcs_path)
            return blob.exists()
        except Exception:
            return False

    def batch_check_screenshots_exist(self, gcs_paths: List[str]) -> Dict[str, bool]:
        """
        Batch check if screenshots exist in GCS using bulk blob operations.

        Args:
            gcs_paths: List of GCS paths (without gs:// prefix)

        Returns:
            Dictionary mapping gcs_path -> exists (True/False)
        """
        if not self._ensure_storage_client():
            return {path: False for path in gcs_paths}

        if not gcs_paths:
            return {}

        result = {}

        try:
            # Create blob objects for all paths
            blobs = [self._bucket.blob(path) for path in gcs_paths]

            # Use batch request to check existence - this is the most efficient approach
            # The exists() method uses HEAD requests which are cheaper than GET
            with ThreadPoolExecutor(max_workers=min(50, len(gcs_paths))) as executor:
                # Submit all existence checks concurrently
                future_to_path = {
                    executor.submit(blob.exists): path
                    for blob, path in zip(blobs, gcs_paths)
                }

                # Collect results
                for future in as_completed(future_to_path):
                    path = future_to_path[future]
                    try:
                        exists = future.result()
                        result[path] = exists
                    except Exception:
                        result[path] = False

            return result

        except Exception as e:
            _print(f"[bold red]Error: Batch existence check failed: {e}")
            # Fall back to individual checks
            for path in gcs_paths:
                try:
                    blob = self._bucket.blob(path)
                    result[path] = blob.exists()
                except Exception:
                    result[path] = False

            return result

    def get_gcs_url(self, gcs_path: str) -> str:
        """
        Get the full GCS URL for a path.

        Args:
            gcs_path: GCS path (without gs:// prefix)

        Returns:
            Full GCS URL (gs://bucket/path)
        """
        return f"gs://{self.bucket_name}/{gcs_path}"

    def get_https_url(self, gcs_path: str) -> str:
        """
        Get the HTTPS URL for a GCS path.

        Args:
            gcs_path: GCS path (without gs:// prefix)

        Returns:
            HTTPS URL for accessing the file
        """
        return f"https://storage.googleapis.com/{self.bucket_name}/{gcs_path}"

    def list_uploaded_screenshots(
        self, dataset_name: str, prefix: Optional[str] = None
    ) -> List[str]:
        """
        List uploaded screenshots for a dataset.

        Args:
            dataset_name: Name of the dataset
            prefix: Optional prefix to filter results

        Returns:
            List of GCS paths for uploaded screenshots
        """
        if not self._ensure_storage_client():
            return []

        try:
            base_prefix = f"screenshots/{dataset_name}/"
            if prefix:
                base_prefix += prefix

            blobs = self._bucket.list_blobs(prefix=base_prefix)
            return [blob.name for blob in blobs if blob.name.endswith(".png")]
        except Exception as e:
            _print(f"[bold red]Error listing screenshots: {e}")
            return []

    def get_upload_stats(
        self, screenshots_dir: Path, dataset_name: str, experiments_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Get statistics about screenshot upload status.

        Args:
            screenshots_dir: Local screenshots directory
            dataset_name: Dataset name
            experiments_df: DataFrame with experiment data

        Returns:
            Dictionary with upload statistics
        """
        upload_tasks = self._prepare_upload_tasks(
            screenshots_dir, dataset_name, experiments_df
        )

        stats = {
            "total_local": len(upload_tasks),
            "uploaded": 0,
            "missing_local": 0,
            "upload_rate": 0.0,
        }

        if not upload_tasks:
            return stats

        # Check how many are already uploaded
        for task in upload_tasks:
            if self.check_screenshot_exists_in_gcs(task["gcs_path"]):
                stats["uploaded"] += 1

        stats["upload_rate"] = (
            stats["uploaded"] / stats["total_local"]
            if stats["total_local"] > 0
            else 0.0
        )

        return stats
