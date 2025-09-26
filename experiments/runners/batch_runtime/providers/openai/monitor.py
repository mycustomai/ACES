import json
import os
from typing import Any, Iterable, Optional

import openai
from openai.types import Batch
from rich import print as _print
from typing_extensions import get_args

from experiments.runners.batch_runtime.typedefs import (BatchStatus,
                                                        BatchStatusResult,
                                                        ProviderBatchId,
                                                        ProviderBatchResult)

from .._base.monitor import BaseBatchProviderMonitor


class OpenAIProviderBatchMonitor(BaseBatchProviderMonitor):
    STATUS_MAPPING = {
        "completed": BatchStatus.COMPLETED,
        "expired": BatchStatus.FAILED,
        "failed": BatchStatus.FAILED,
        "cancelled": BatchStatus.FAILED,
        "cancelling": BatchStatus.FAILED,
        "in_progress": BatchStatus.IN_PROGRESS,
        "validating": BatchStatus.IN_PROGRESS,
        "finalizing": BatchStatus.IN_PROGRESS,
    }

    def _setup(self):
        self._validate_mapping()

        self.api_key = self.engine_params.api_key or os.getenv("OPENAI_API_KEY")

        if not self.api_key:
            _print("[bold red]No OpenAI API key configured. Cannot continue.")
            _print(
                "[bold red]Set OPENAI_API_KEY environment variable or utilize `EngineParam.api_key`."
            )
            raise ValueError("No OpenAI API key configured.")

        self.client = openai.OpenAI(api_key=self.api_key)

    def monitor_batches(
        self, batches: list[ProviderBatchId]
    ) -> list[BatchStatusResult]:
        """Monitor and retrieve batches if possible"""
        if not self.client:
            raise ValueError("OpenAI client not configured. Cannot monitor batches.")

        results = []

        for batch_id in batches:
            batch_status_info = self._check_batch_status(batch_id)
            if not batch_status_info:
                continue

            openai_status = batch_status_info.status
            status = self._map_openai_status(openai_status)

            result = None
            if status == BatchStatus.COMPLETED:
                result = self._accumulate_batch_results(batch_id, batch_status_info)

            results.append(
                BatchStatusResult(batch_id=batch_id, status=status, result=result)
            )

        return results

    def _check_batch_status(self, batch_id: str) -> Optional[Batch]:
        """Check the status of a submitted batch."""
        return self.client.batches.retrieve(batch_id)

    @classmethod
    def _map_openai_status(cls, openai_status: str) -> BatchStatus:
        """Map OpenAI batch status to the common BatchStatus enum."""
        mapped = cls.STATUS_MAPPING.get(openai_status)
        if not mapped:
            _print(f"[yellow]Unknown OpenAI batch status: {openai_status}")
            _print(f"[yellow]Using default status: {BatchStatus.FAILED}")
            mapped = BatchStatus.FAILED
        return mapped

    def _accumulate_batch_results(
        self, batch_id: str, batch_status_info: Batch
    ) -> Optional[ProviderBatchResult]:
        """Download batch results for a completed batch."""
        # get normal output
        output_file_id = batch_status_info.output_file_id
        if not output_file_id:
            raise ValueError(
                f"No output file ID found for batch {batch_id}. Unknown error. Cannot download results."
            )

        results = list(self._download_jsonl(output_file_id))

        error_file_id = batch_status_info.error_file_id
        if error_file_id:
            results.extend(self._download_jsonl(error_file_id))

        return ProviderBatchResult({"results": results})

    def _download_jsonl(self, file_id: str) -> Iterable[dict[str, Any]]:
        """Download batch results for a completed batch."""
        file_response = self.client.files.content(file_id)
        content = file_response.content.decode("utf-8")

        for line in content.strip().split("\n"):
            if line.strip():
                yield json.loads(line)

    @classmethod
    def _validate_mapping(cls) -> None:
        """Introspect the OpenAI status mapping and validate that all declared statuses are mapped.

        Returns:
            None if `STATUS_MAPPING` covers all statuses, otherwise raises an exception.

        Note:
            - Refer to `openai.types.Batch.status` for a list of valid statuses.
            - If `STATUS_MAPPING` includes extra statuses, a warning text will be printed.
        """
        allowed_statuses = set(get_args(Batch.__annotations__["status"]))
        missing = allowed_statuses - set(cls.STATUS_MAPPING.keys())
        if missing:
            raise ValueError(f"Missing STATUS_MAPPING entries for statuses: {missing}")
        extra = set(cls.STATUS_MAPPING.keys()) - allowed_statuses
        if extra:
            _print(
                f"[yellow dim]Extra STATUS_MAPPING entries for unknown statuses: {extra}"
            )
