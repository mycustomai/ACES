import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, List

from google.cloud import storage
from pydantic import BaseModel
from rich import print as _print
from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn

from experiments.config import ExperimentData
from experiments.utils.dataset_ops import get_dataset_name


class _UploadTask(BaseModel):
    local_path: str
    gcs_path: str


class GCSManager:
    """For managing screenshots on GCS.

    Attributes:
        max_workers: Number of concurrent upload threads.

    """

    def __init__(
        self,
        local_dataset_path: str,
        max_workers: int = 16,
    ):
        # TODO: this needs to be self-contained in a dataset manager class so as to decouple local dataset vs hf dataset.
        #  this class may be passed in on object initialization
        self.dataset_name = get_dataset_name(local_dataset_path)
        dataset_dir = Path(local_dataset_path).parent
        self.screenshots_dir = dataset_dir / "screenshots"
        self.local_dataset_path = local_dataset_path

        self.max_workers = max_workers

        self.bucket_name = os.getenv("GCS_BUCKET_NAME")
        self.project_id = os.getenv("GOOGLE_CLOUD_PROJECT")

        if not self.bucket_name:
            raise ValueError(
                "GCS bucket name must be provided or set in GCS_BUCKET_NAME environment variable"
            )

        self._storage_client = storage.Client(project=self.project_id)
        self._bucket = self._storage_client.bucket(self.bucket_name)

    def upload(
        self, experiments: Iterable[List[ExperimentData]], verbose: bool = False
    ):
        uploaded_screenshot_paths = set(self._list_uploaded_screenshots())
        if verbose:
            _print(
                f"[dim]Detected {len(uploaded_screenshot_paths)} screenshots already uploaded to GCS"
            )

        not_uploaded: list[ExperimentData] = []

        for experiment in experiments:
            for data in experiment:
                if data.get_gcs_screenshot_path() not in uploaded_screenshot_paths:
                    not_uploaded.append(data)

        if verbose:
            _print(f"[dim]Uploading {len(not_uploaded)} new screenshots to GCS")

        upload_tasks = self._prepare_upload_tasks(not_uploaded)
        self._upload_screenshots(upload_tasks)

    def _prepare_upload_tasks(
        self, experiments: list[ExperimentData]
    ) -> list[_UploadTask]:
        tasks: list[_UploadTask] = []
        for data in experiments:
            # TODO: create temp path if using hf dataset
            experiments_path = data.get_local_screenshot_path(self.screenshots_dir)
            if not experiments_path.exists():
                raise FileNotFoundError(
                    f"Screenshot file not found: {experiments_path}"
                )
            else:
                gcs_path = data.get_gcs_screenshot_path()
                tasks.append(
                    _UploadTask(local_path=str(experiments_path), gcs_path=gcs_path)
                )
        return tasks

    def _upload_single_screenshot(self, task: _UploadTask):
        """
        Upload a single screenshot task.

        Returns:
            True if upload was successful
        """
        blob = self._bucket.blob(task.gcs_path)
        blob.upload_from_filename(task.local_path)

    def _upload_screenshots(self, upload_tasks: list[_UploadTask]):
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
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_task = {
                    executor.submit(self._upload_single_screenshot, task): task
                    for task in upload_tasks
                }

                # Process completed uploads
                for future in as_completed(future_to_task):
                    future.result()
                    progress.advance(task_id)

    def _list_uploaded_screenshots(self) -> List[str]:
        """List all screenshots uploaded to GCS.

        Note:
            The underlying `list_blobs` method has no limit on the number of results.

        Returns:
            The list of GCS paths for uploaded screenshots.

            Example:
            ```
            screenshots/bias_experiments/bidet/overall_pick_bias/bidet_overall_pick_bias_0.png
            screenshots/bias_experiments/washing_machine/sponsored_tag_bias/washing_machine_sponsored_tag_bias_997.png
            ...
            ```

        """
        base_prefix = f"screenshots/{self.dataset_name}"
        blobs = self._bucket.list_blobs(prefix=base_prefix)
        return [blob.name for blob in blobs]
