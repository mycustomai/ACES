import os
from typing import Any, Iterable, Optional, get_args

import anthropic
from anthropic.types.beta.messages import (BetaMessageBatch,
                                           BetaMessageBatchIndividualResponse)
from rich import print as _print

from experiments.runners.batch_new.typedefs import (BatchStatus,
                                                    BatchStatusResult,
                                                    ProviderBatchId,
                                                    ProviderBatchResult)

from .._base.monitor import BaseBatchProviderMonitor


class AnthropicBatchProviderMonitor(BaseBatchProviderMonitor):
    STATUS_MAPPING = {
        "in_progress": BatchStatus.IN_PROGRESS,
        "ended": BatchStatus.COMPLETED,
        "canceling": BatchStatus.FAILED,
    }

    def _setup(self):
        """Initialize Anthropic client and validate configuration."""
        self._validate_mapping()

        self.api_key = self.engine_params.api_key or os.getenv("ANTHROPIC_API_KEY")

        if not self.api_key:
            _print("[bold red]No Anthropic API key configured. Cannot continue.")
            _print(
                "[bold red]Set ANTHROPIC_API_KEY environment variable or utilize `EngineParam.api_key`."
            )
            raise ValueError("No Anthropic API key configured.")

        self.client = anthropic.Anthropic(api_key=self.api_key)

    def monitor_batches(
        self, batches: list[ProviderBatchId]
    ) -> list[BatchStatusResult]:
        """Monitor and retrieve batches if possible"""
        if not self.client:
            raise ValueError("Anthropic client not configured. Cannot monitor batches.")

        results = []

        for batch_id in batches:
            batch_status_info: BetaMessageBatch = (
                self.client.beta.messages.batches.retrieve(batch_id)
            )
            if not batch_status_info:
                raise ValueError(f"Batch {batch_id} not found.")

            anthropic_status = batch_status_info.processing_status
            status = self._map_anthropic_status(anthropic_status)

            result = None
            if status == BatchStatus.COMPLETED:
                result = self._accumulate_batch_results(batch_id)

            results.append(
                BatchStatusResult(batch_id=batch_id, status=status, result=result)
            )

        return results

    @classmethod
    def _map_anthropic_status(cls, anthropic_status: str) -> BatchStatus:
        """Map Anthropic batch status to the common BatchStatus enum."""
        mapped = cls.STATUS_MAPPING.get(anthropic_status)
        if not mapped:
            _print(f"[yellow]Unknown Anthropic batch status: {anthropic_status}")
            _print(f"[yellow]Using default status: {BatchStatus.FAILED}")
            mapped = BatchStatus.FAILED
        return mapped

    def _accumulate_batch_results(self, batch_id: str) -> Optional[ProviderBatchResult]:
        """Download batch results for a completed batch."""
        try:
            results_generator: Iterable[BetaMessageBatchIndividualResponse] = (
                self.client.beta.messages.batches.results(batch_id)
            )
        except Exception as e:
            _print(
                f"[bold yellow]Error: Failed to download results for batch {batch_id}: {e}"
            )
            raise e

        try:
            results = list(self._process_batch_results(results_generator))
            return ProviderBatchResult({"results": results})
        except Exception as e:
            _print(
                f"[bold yellow]Error: Failed to process results for batch {batch_id}: {e}"
            )
            raise e

    @staticmethod
    def _process_batch_results(
        results_generator: Iterable[BetaMessageBatchIndividualResponse],
    ) -> Iterable[dict[str, Any]]:
        """Process batch results from Anthropic generator.

        `BetaMessageBatchIndividualResponse` is a union whose discriminator is `type`.
        """
        for result in results_generator:
            try:
                yield result.model_dump()
            except AttributeError:
                raise ValueError(
                    f"Non-pydantic type received as batch result line: {type(result)}"
                )

    @classmethod
    def _validate_mapping(cls) -> None:
        """Introspect the Anthropic status mapping and validate that all declared statuses are mapped.

        Returns:
            None if `STATUS_MAPPING` covers all statuses, otherwise raises an exception.

        Note:
            - Refer to `openai.types.Batch.status` for a list of valid statuses.
            - If `STATUS_MAPPING` includes extra statuses, a warning text will be printed.
        """
        allowed_statuses = set(
            get_args(BetaMessageBatch.__annotations__["processing_status"])
        )
        missing = allowed_statuses - set(cls.STATUS_MAPPING.keys())
        if missing:
            raise ValueError(f"Missing STATUS_MAPPING entries for statuses: {missing}")
        extra = set(cls.STATUS_MAPPING.keys()) - allowed_statuses
        if extra:
            _print(
                f"[yellow dim]Extra STATUS_MAPPING entries for unknown statuses: {extra}"
            )
