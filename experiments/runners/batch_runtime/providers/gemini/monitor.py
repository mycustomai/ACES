import json
import os
from typing import Optional

from google import genai
from google.cloud import storage
from google.genai.types import BatchJob, HttpOptions, JobState, ListBatchJobsConfig
from rich import print as _print

from experiments.runners.batch_runtime.typedefs import (
    BatchStatus,
    BatchStatusResult,
    ProviderBatchId,
    ProviderBatchResult,
)

from .._base.monitor import BaseBatchProviderMonitor


class GeminiProviderBatchMonitor(BaseBatchProviderMonitor):
    """Monitor Gemini batch jobs and retrieve results from GCS."""

    STATUS_MAPPING = {
        JobState.JOB_STATE_SUCCEEDED: BatchStatus.COMPLETED,
        JobState.JOB_STATE_FAILED: BatchStatus.FAILED,
        JobState.JOB_STATE_CANCELLED: BatchStatus.FAILED,
        JobState.JOB_STATE_EXPIRED: BatchStatus.FAILED,
        JobState.JOB_STATE_CANCELLING: BatchStatus.FAILED,
        JobState.JOB_STATE_RUNNING: BatchStatus.IN_PROGRESS,
        JobState.JOB_STATE_PENDING: BatchStatus.IN_PROGRESS,
        JobState.JOB_STATE_PAUSED: BatchStatus.IN_PROGRESS,
        JobState.JOB_STATE_UNSPECIFIED: BatchStatus.IN_PROGRESS,
        JobState.JOB_STATE_QUEUED: BatchStatus.IN_PROGRESS,
        JobState.JOB_STATE_UPDATING: BatchStatus.IN_PROGRESS,
        JobState.JOB_STATE_PARTIALLY_SUCCEEDED: BatchStatus.IN_PROGRESS,
    }

    def _setup(self):
        """Initialize Gemini/Vertex AI client and validate configuration."""
        self.api_key = self.engine_params.api_key or os.getenv("GOOGLE_API_KEY")
        self.project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        self.bucket_name = os.getenv("GCS_BUCKET_NAME")

        # Validate configuration
        errors = []
        if not self.api_key:
            errors.append("GOOGLE_API_KEY not set")
        if not self.project_id:
            errors.append("GOOGLE_CLOUD_PROJECT not set")
        if not self.bucket_name:
            errors.append("GCS_BUCKET_NAME not set")

        if errors:
            _print(f"[bold red]Missing Google Cloud configuration: {', '.join(errors)}")
            _print(
                "[bold red]Set required environment variables or utilize EngineParams."
            )
            raise ValueError(f"Missing configuration: {', '.join(errors)}")

        self._init_clients()

        self._batch_output_uri_mappings: dict[str, str] = {}

    def _init_clients(self) -> None:
        """Initialize Google Cloud clients."""
        self.genai_client = genai.Client(
            http_options=HttpOptions(api_version="v1"),
            vertexai=True,
            location="global",
        )

        self.storage_client = storage.Client(project=self.project_id)
        self.bucket = self.storage_client.bucket(self.bucket_name)

        _print("[bold green]✓ Initialized Google Cloud clients for monitoring")

    def monitor_batches(
        self, batches: list[ProviderBatchId]
    ) -> list[BatchStatusResult]:
        """Monitor and retrieve batches if possible using bulk monitoring for efficiency."""
        if not hasattr(self, "genai_client"):
            raise ValueError("Gemini client not configured. Cannot monitor batches.")

        results = []

        target_batch_ids = set(batches)

        # use bulk monitor
        batch_list = self._list_recent_batches()
        for batch_info in batch_list:
            batch_name = ProviderBatchId(batch_info.name)
            if not batch_name:
                raise ValueError(f"Batch has no name: {batch_name}")
            if batch_name in target_batch_ids:
                # Map JobState to BatchStatus
                status = self._map_gemini_status(batch_info.state)

                result = None
                if status == BatchStatus.COMPLETED:
                    self._batch_output_uri_mappings[batch_name] = (
                        batch_info.dest.gcs_uri
                    )

                    # Download results from GCS
                    result = self._download_batch_results(batch_name)

                results.append(
                    BatchStatusResult(batch_id=batch_name, status=status, result=result)
                )

                target_batch_ids.remove(batch_name)

                if not target_batch_ids:
                    break

        # fallback to single-batch checking
        for batch_id in target_batch_ids:
            status_result = self._check_single_batch(batch_id)
            if status_result:
                results.append(status_result)

        return results

    def _list_recent_batches(self) -> list[BatchJob]:
        """List recent batch jobs using the genai client."""
        batch_list = []

        for job in self.genai_client.batches.list(
            config=ListBatchJobsConfig(page_size=100)
        ):
            batch_list.append(job)
            if len(batch_list) >= 200:
                break

        return batch_list

    def _check_single_batch(self, batch_id: str) -> Optional[BatchStatusResult]:
        """Check the status of a single batch."""
        try:
            job = self.genai_client.batches.get(name=batch_id)
            status = self._map_gemini_status(job.state)

            result = None
            if status == BatchStatus.COMPLETED:
                if job.dest is None:
                    raise ValueError(f"Batch {batch_id} has no output directory URI.")
                self._batch_output_uri_mappings[batch_id] = job.dest.gcs_uri

                result = self._download_batch_results(batch_id)

            return BatchStatusResult(
                batch_id=ProviderBatchId(batch_id),
                result=result,
                status=status,
            )

        except Exception as e:
            _print(f"[bold red]Failed to check batch {batch_id}: {e}")
            return None

    @classmethod
    def _map_gemini_status(cls, gemini_state: JobState) -> BatchStatus:
        """Map Gemini JobState to the common BatchStatus enum."""
        mapped = cls.STATUS_MAPPING.get(gemini_state)
        if not mapped:
            _print(f"[yellow]Unknown Gemini job state: {gemini_state}")
            _print(f"[yellow]Using default status: {BatchStatus.FAILED}")
            mapped = BatchStatus.FAILED
        return mapped

    def _download_batch_results(self, batch_id: str) -> Optional[ProviderBatchResult]:
        """Download batch results from GCS."""
        # Get output directory URI
        output_dir_uri = self._batch_output_uri_mappings.get(batch_id)

        if not output_dir_uri:
            raise ValueError(f"Batch {batch_id} has no output directory URI.")

        if not output_dir_uri.startswith("gs://"):
            raise ValueError(f"Invalid GCS URI format: {output_dir_uri}")

        path_parts = output_dir_uri.replace("gs://", "").split("/", 1)
        if len(path_parts) != 2:
            raise ValueError(f"[bold red]Invalid GCS URI format: {output_dir_uri}")

        bucket_name, prefix = path_parts

        # Get bucket and list blobs
        bucket = self.storage_client.bucket(bucket_name)
        blobs = list(bucket.list_blobs(prefix=prefix))

        # Look for result files (predictions.jsonl or other JSONL files)
        result_blobs = []
        for blob in blobs:
            if blob.size > 0 and blob.name.endswith(".jsonl"):
                result_blobs.append(blob)

        if not result_blobs:
            raise ValueError(f"[bold yellow]No result files found in {output_dir_uri}")

        # Download and parse results
        all_results = []
        for blob in result_blobs:
            content = blob.download_as_text()
            for line in content.strip().split("\n"):
                if line:
                    try:
                        result = json.loads(line)
                        all_results.append(result)
                    except json.JSONDecodeError:
                        continue

        return ProviderBatchResult({"results": all_results})
