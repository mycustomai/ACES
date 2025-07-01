import json
import os
from io import BytesIO

import backoff
import openai
from rich import print as _print

from experiments.runners.batch_new.typedefs import (ExperimentSubmissionRecord,
                                                    ProviderBatchId,
                                                    ProviderBatchRequest,
                                                    SerializedBatchRequest)

from .._base.submit import BaseBatchProviderSubmitter


class OpenAIBatchProviderSubmitter(BaseBatchProviderSubmitter):
    def _setup(self):
        self.api_key = self.engine_params.api_key or os.getenv("OPENAI_API_KEY")

        if not self.api_key:
            _print(
                "[bold yellow]Warning: No OpenAI API key configured. Batch submission will fail."
            )
            _print(
                "[bold yellow]Set OPENAI_API_KEY environment variable or utilize `EngineParam.api_key`."
            )
            raise ValueError("No OpenAI API key configured.")

        self.client = openai.OpenAI(api_key=self.api_key)

    def submit(
        self, requests: list[SerializedBatchRequest], verbose: bool = False
    ) -> list[ExperimentSubmissionRecord]:
        """Sends request and returns provider-specific Batch Id"""
        if not self.client:
            raise ValueError("OpenAI client not configured. Cannot submit batch.")

        # Chunk the complete serialized requests
        chunks: list[list[SerializedBatchRequest]] = self.chunk_requests(
            requests,
            self.DEFAULT_CHUNKING_STRATEGY,
            max_size_mb=self.DEFAULT_MAX_SIZE_MB,
        )

        submission_records = []

        for chunk_index, chunk in enumerate(chunks):
            # Extract provider requests for upload
            provider_requests = [req.provider_request for req in chunk]
            file_id = self._upload_batch_file(provider_requests, chunk_index)
            batch_id = self._submit_batch_to_api(file_id, chunk_index)

            # Create submission records for this chunk
            for serialized_request in chunk:
                submission_records.append(
                    ExperimentSubmissionRecord(
                        experiment_id=serialized_request.experiment_id,
                        batch_id=ProviderBatchId(batch_id),
                        config_name=self.engine_params.config_name,
                    )
                )

        return submission_records

    @backoff.on_exception(backoff.constant, Exception, interval=1, max_tries=5)
    def _upload_batch(
        self, batch_data: list[ProviderBatchRequest], batch_index: int
    ) -> ProviderBatchId:
        file_id = self._upload_batch_file(batch_data, batch_index)
        batch_id = self._submit_batch_to_api(file_id, batch_index)

        return batch_id

    def _upload_batch_file(
        self, batch_requests: list[ProviderBatchRequest], chunk_index: int
    ) -> str:
        """Upload a batch file to OpenAI and return the file ID."""
        if not self.client:
            raise ValueError("OpenAI client not configured. Cannot upload batch file.")

        # Create JSONL content
        jsonl_content = "\n".join(json.dumps(request) for request in batch_requests)

        chunk_name = f"batch_chunk_{chunk_index}"
        _print(
            f"[bold blue]Uploading OpenAI batch file '{chunk_name}' with {len(batch_requests)} requests..."
        )

        file_obj = BytesIO(jsonl_content.encode("utf-8"))
        file_obj.name = f"{chunk_name}.jsonl"

        file_response = self.client.files.create(file=file_obj, purpose="batch")

        file_id = file_response.id
        _print(
            f"[bold green]✓ OpenAI file '{chunk_name}' uploaded successfully. File ID: {file_id}"
        )
        return file_id

    def _submit_batch_to_api(self, file_id: str, chunk_index: int) -> ProviderBatchId:
        """Submit a batch job to OpenAI using an uploaded file."""
        if not self.client:
            raise ValueError("OpenAI client not configured. Cannot submit batch.")

        chunk_name = f"batch_chunk_{chunk_index}"
        _print(f"[bold blue]Submitting batch '{chunk_name}' to OpenAI API...")

        batch = self.client.batches.create(
            input_file_id=file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )

        batch_id = batch.id
        _print(
            f"[bold green]✓ OpenAI batch '{chunk_name}' submitted successfully. Batch ID: {batch_id}"
        )
        return ProviderBatchId(batch_id)
