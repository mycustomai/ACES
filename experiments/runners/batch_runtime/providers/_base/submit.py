import json
import sys
from abc import abstractmethod
from enum import Enum
from typing import List

from experiments.runners.batch_runtime.typedefs import (ExperimentSubmissionRecord,
                                                        SerializedBatchRequest)

from ._base import BaseBatchProvider


class ChunkingStrategy(Enum):
    """Strategies for chunking batch requests."""

    BY_COUNT = "by_count"
    BY_SIZE = "by_size"


class ChunkingMixin:
    """Mixin providing request chunking strategies."""

    @staticmethod
    def chunk_by_count(
        requests: List[SerializedBatchRequest], chunk_size: int
    ) -> List[List[SerializedBatchRequest]]:
        """
        Chunk requests by count.

        Args:
            requests: List of serialized batch requests
            chunk_size: Maximum requests per chunk

        Returns:
            List of request chunks
        """
        chunks = []
        for i in range(0, len(requests), chunk_size):
            chunk = requests[i : i + chunk_size]
            chunks.append(chunk)
        return chunks

    @staticmethod
    def chunk_by_size(
        requests: List[SerializedBatchRequest], max_size_mb: int
    ) -> List[List[SerializedBatchRequest]]:
        """
        Chunk requests by total size.

        Args:
            requests: List of serialized batch requests
            max_size_mb: Maximum size per chunk in MB

        Returns:
            List of request chunks
        """
        max_size_bytes = max_size_mb * 1024 * 1024
        chunks = []
        current_chunk = []
        current_size = 0

        for request in requests:
            try:
                request_size = len(json.dumps(request.provider_request).encode("utf-8"))
            except Exception:
                # Fallback to approximate size calculation if JSON serialization fails
                request_size = sys.getsizeof(request.provider_request)

            if current_size + request_size > max_size_bytes and current_chunk:
                # Start new chunk
                chunks.append(current_chunk)
                current_chunk = [request]
                current_size = request_size
            else:
                current_chunk.append(request)
                current_size += request_size

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def chunk_requests(
        self,
        requests: List[SerializedBatchRequest],
        strategy: ChunkingStrategy,
        **kwargs,
    ) -> List[List[SerializedBatchRequest]]:
        """
        Chunk requests using specified strategy.

        Args:
            requests: List of serialized batch requests
            strategy: Chunking strategy to use
            **kwargs: Strategy-specific parameters

        Returns:
            List of request chunks
        """
        if strategy == ChunkingStrategy.BY_COUNT:
            chunk_size = kwargs.get("chunk_size", 1000)
            return self.chunk_by_count(requests, chunk_size)
        elif strategy == ChunkingStrategy.BY_SIZE:
            max_size_mb = kwargs.get("max_size_mb", 100)
            return self.chunk_by_size(requests, max_size_mb)
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")


class BaseBatchProviderSubmitter(BaseBatchProvider, ChunkingMixin):
    """Base class for batch providers with chunking capabilities."""

    # Default chunking configuration - can be overridden by subclasses
    DEFAULT_CHUNKING_STRATEGY = ChunkingStrategy.BY_COUNT
    DEFAULT_CHUNK_SIZE = 1000
    DEFAULT_MAX_SIZE_MB = 100

    @abstractmethod
    def submit(
        self, requests: list[SerializedBatchRequest]
    ) -> list[ExperimentSubmissionRecord]:
        """Sends a request and returns provider-specific ``BatchId``"""
        ...
