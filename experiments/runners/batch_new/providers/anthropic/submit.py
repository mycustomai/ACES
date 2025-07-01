import os

import anthropic
import backoff
from anthropic.types.messages.batch_create_params import Request
from rich import print as _print

from experiments.runners.batch_new.typedefs import (ExperimentSubmissionRecord,
                                                    ProviderBatchId,
                                                    SerializedBatchRequest)

from .._base.submit import BaseBatchProviderSubmitter


class AnthropicBatchProviderSubmitter(BaseBatchProviderSubmitter):
    def _setup(self) -> None:
        """Initialize Anthropic client and validate configuration."""
        self.api_key = self.engine_params.api_key or os.getenv("ANTHROPIC_API_KEY")

        if not self.api_key:
            _print(
                "[bold yellow]Warning: No Anthropic API key configured. Batch submission will fail."
            )
            _print(
                "[bold yellow]Set ANTHROPIC_API_KEY environment variable or utilize `EngineParams.api_key`."
            )
            raise ValueError("No Anthropic API key configured.")

        self.client: anthropic.Anthropic = anthropic.Anthropic(api_key=self.api_key)

    def submit(
        self, requests: list[SerializedBatchRequest]
    ) -> list[ExperimentSubmissionRecord]:
        """Submit batch requests to Anthropic API and return submission records."""
        if not self.client:
            raise ValueError("Anthropic client not configured. Cannot submit batch.")

        chunks = self.chunk_requests(
            requests, self.DEFAULT_CHUNKING_STRATEGY, chunk_size=self.DEFAULT_CHUNK_SIZE
        )

        submission_records = []

        for chunk in chunks:
            provider_requests = [req.provider_request for req in chunk]

            batch_id = self._submit_chunk_to_api(provider_requests)

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
    def _submit_chunk_to_api(self, provider_requests: list[Request]) -> str:
        """Submit a single chunk of requests to Anthropic API with retry logic."""
        if not self.client:
            raise ValueError("Anthropic client not configured. Cannot submit batch.")

        # noinspection PyTypeChecker
        response = self.client.beta.messages.batches.create(requests=provider_requests)

        return response.id
