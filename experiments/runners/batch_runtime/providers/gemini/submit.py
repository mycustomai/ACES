import json
import os
import time

from google import genai
from google.cloud import storage
from google.genai.types import CreateBatchJobConfig, HttpOptions
from rich import print as _print

from experiments.runners.batch_runtime.typedefs import (
    ExperimentSubmissionRecord,
    ProviderBatchId,
    SerializedBatchRequest,
)

from .._base.submit import BaseBatchProviderSubmitter


class GeminiBatchProviderSubmitter(BaseBatchProviderSubmitter):
    """Gemini batch provider submitter with GCS integration."""

    def _setup(self) -> None:
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
            _print(
                f"[bold yellow]Warning: Missing Google Cloud configuration: {', '.join(errors)}"
            )
            _print(
                "[bold yellow]Set required environment variables or utilize EngineParams."
            )
            raise ValueError(f"Missing configuration: {', '.join(errors)}")

        self._init_clients()

    def _init_clients(self) -> None:
        """Initialize Google Cloud clients."""

        self.genai_client = genai.Client(
            http_options=HttpOptions(api_version="v1"),
            location="global"
        )
        self.storage_client = storage.Client(project=self.project_id)
        self.bucket = self.storage_client.bucket(self.bucket_name)

    def submit(
        self, requests: list[SerializedBatchRequest]
    ) -> list[ExperimentSubmissionRecord]:
        """Submit batch requests to Gemini API via GCS and return submission records."""
        if not hasattr(self, "genai_client"):
            raise ValueError("Gemini client not configured. Cannot submit batch.")

        chunks = self.chunk_requests(
            requests, self.DEFAULT_CHUNKING_STRATEGY, chunk_size=self.DEFAULT_CHUNK_SIZE
        )

        submission_records = []

        date_sent = time.strftime("%Y-%m-%d-%H:%M:%S")

        for chunk_index, chunk in enumerate(chunks):
            batch_id = self._submit_chunk(chunk, date_sent, chunk_index)

            for serialized_request in chunk:
                submission_records.append(
                    ExperimentSubmissionRecord(
                        experiment_id=serialized_request.experiment_id,
                        batch_id=ProviderBatchId(batch_id),
                        config_name=self.engine_params.config_name,
                    )
                )

        return submission_records

    def _submit_chunk(
        self, chunk: list[SerializedBatchRequest], date_sent: str, chunk_index: int
    ) -> str:
        """Submit a single chunk of requests to Gemini API."""
        chunk_name = (
            f"gemini_batch_{self.engine_params.config_name}_chunk_{chunk_index}"
        )

        provider_requests = [req.provider_request for req in chunk]

        jsonl_content = "\n".join(json.dumps(req) for req in provider_requests)

        gcs_input_path = f"batch_inputs/{chunk_name}_{date_sent}.jsonl"
        blob = self.bucket.blob(gcs_input_path)
        blob.upload_from_string(jsonl_content)
        gcs_input_uri = f"gs://{self.bucket_name}/{gcs_input_path}"

        model_id = self.engine_params.model
        model_resource_name = f"publishers/google/models/{model_id}"

        gcs_output_uri = (
            f"gs://{self.bucket_name}/batch_outputs/{chunk_name}_{date_sent}/"
        )

        batch_job = self.genai_client.batches.create(
            model=model_resource_name,
            src=gcs_input_uri,
            config=CreateBatchJobConfig(dest=gcs_output_uri),
        )

        return batch_job.name
