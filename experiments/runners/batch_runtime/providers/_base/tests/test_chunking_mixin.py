from unittest.mock import Mock, patch

import pandas as pd
import pytest

from common.messages import RawMessageExchange
from experiments.config import ExperimentData
from experiments.runners.batch_runtime.providers._base.submit import (
    ChunkingMixin, ChunkingStrategy)
from experiments.runners.batch_runtime.typedefs import (ProviderBatchRequest,
                                                        SerializedBatchRequest)


class TestChunkingMixin:
    @pytest.fixture
    def chunking_mixin(self):
        """Create a ChunkingMixin instance for testing."""
        return ChunkingMixin()

    @pytest.fixture
    def mock_experiment_data(self):
        """Create mock ExperimentData objects."""
        mock_df = pd.DataFrame(
            {
                "product_name": ["Product A", "Product B", "Product C"],
                "price": [10.0, 20.0, 30.0],
                "description": [
                    "Short desc",
                    "Medium description text",
                    "Very long description with lots of text",
                ],
            }
        )

        experiments = []
        for i in range(3):
            exp = ExperimentData(
                experiment_df=mock_df.iloc[i : i + 1],
                query="test query",
                experiment_label=f"test_label_{i}",
                experiment_number=i,
                dataset_name="test_dataset",
                prompt_template="test prompt template",
            )
            experiments.append(exp)

        return experiments

    @pytest.fixture
    def mock_raw_messages(self):
        """Create mock RawMessageExchange objects."""
        messages = []
        for i in range(3):
            msg = RawMessageExchange([{"role": "user", "content": f"Test message {i}"}])
            messages.append(msg)

        return messages

    @pytest.fixture
    def sample_serialized_batch_requests(self, mock_experiment_data, mock_raw_messages):
        """Create sample SerializedBatchRequest objects for testing."""
        serialized_requests = []
        for i in range(len(mock_experiment_data)):
            provider_req = ProviderBatchRequest(
                {
                    "custom_id": f"test_id_{i}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": "gpt-4",
                        "messages": mock_raw_messages[i],
                        "max_tokens": 1000,
                        "temperature": 0.7,
                    },
                }
            )
            serialized_req = SerializedBatchRequest(
                experiment_id=mock_experiment_data[i].experiment_id,
                provider_request=provider_req,
            )
            serialized_requests.append(serialized_req)

        return serialized_requests

    def test_chunk_by_count_basic(
        self, chunking_mixin, sample_serialized_batch_requests
    ):
        """Test basic chunking by count."""
        chunk_size = 2
        chunks = chunking_mixin.chunk_by_count(
            sample_serialized_batch_requests, chunk_size
        )

        assert len(chunks) == 2
        assert len(chunks[0]) == 2
        assert len(chunks[1]) == 1

        # Verify that all original requests are present
        all_requests = [req for chunk in chunks for req in chunk]
        assert len(all_requests) == len(sample_serialized_batch_requests)

    def test_chunk_by_count_exact_division(
        self, chunking_mixin, sample_serialized_batch_requests
    ):
        """Test chunking by count with exact division."""
        chunk_size = 3
        chunks = chunking_mixin.chunk_by_count(
            sample_serialized_batch_requests, chunk_size
        )

        assert len(chunks) == 1
        assert len(chunks[0]) == 3

    def test_chunk_by_count_larger_than_input(
        self, chunking_mixin, sample_serialized_batch_requests
    ):
        """Test chunking by count with chunk size larger than input."""
        chunk_size = 10
        chunks = chunking_mixin.chunk_by_count(
            sample_serialized_batch_requests, chunk_size
        )

        assert len(chunks) == 1
        assert len(chunks[0]) == 3

    def test_chunk_by_count_empty_input(self, chunking_mixin):
        """Test chunking by count with empty input."""
        chunks = chunking_mixin.chunk_by_count([], 5)

        assert len(chunks) == 0

    def test_chunk_by_size_basic(
        self, chunking_mixin, sample_serialized_batch_requests
    ):
        """Test basic chunking by size."""
        # Use a very small max size to force multiple chunks
        max_size_mb = 0.001  # 1KB
        chunks = chunking_mixin.chunk_by_size(
            sample_serialized_batch_requests, max_size_mb
        )

        # Should create multiple chunks due to small size limit
        assert len(chunks) >= 1

        # Verify that all original requests are present
        all_requests = [req for chunk in chunks for req in chunk]
        assert len(all_requests) == len(sample_serialized_batch_requests)

    def test_chunk_by_size_large_limit(
        self, chunking_mixin, sample_serialized_batch_requests
    ):
        """Test chunking by size with large size limit."""
        max_size_mb = 100  # 100MB - should fit all requests
        chunks = chunking_mixin.chunk_by_size(
            sample_serialized_batch_requests, max_size_mb
        )

        # Should create only one chunk
        assert len(chunks) == 1
        assert len(chunks[0]) == 3

    def test_chunk_by_size_empty_input(self, chunking_mixin):
        """Test chunking by size with empty input."""
        chunks = chunking_mixin.chunk_by_size([], 1)

        assert len(chunks) == 0

    def test_chunk_requests_by_count_strategy(
        self, chunking_mixin, sample_serialized_batch_requests
    ):
        """Test chunk_requests method with BY_COUNT strategy."""
        chunks = chunking_mixin.chunk_requests(
            sample_serialized_batch_requests, ChunkingStrategy.BY_COUNT, chunk_size=2
        )

        assert len(chunks) == 2
        assert len(chunks[0]) == 2
        assert len(chunks[1]) == 1

    def test_chunk_requests_by_size_strategy(
        self, chunking_mixin, sample_serialized_batch_requests
    ):
        """Test chunk_requests method with BY_SIZE strategy."""
        chunks = chunking_mixin.chunk_requests(
            sample_serialized_batch_requests, ChunkingStrategy.BY_SIZE, max_size_mb=100
        )

        assert len(chunks) == 1
        assert len(chunks[0]) == 3

    def test_chunk_requests_default_parameters(
        self, chunking_mixin, sample_serialized_batch_requests
    ):
        """Test chunk_requests method with default parameters."""
        # Test default chunk_size for BY_COUNT
        chunks = chunking_mixin.chunk_requests(
            sample_serialized_batch_requests, ChunkingStrategy.BY_COUNT
        )

        # With default chunk_size of 1000, should be one chunk
        assert len(chunks) == 1
        assert len(chunks[0]) == 3

        # Test default max_size_mb for BY_SIZE
        chunks = chunking_mixin.chunk_requests(
            sample_serialized_batch_requests, ChunkingStrategy.BY_SIZE
        )

        # With default max_size_mb of 100, should be one chunk
        assert len(chunks) == 1
        assert len(chunks[0]) == 3

    def test_chunk_requests_invalid_strategy(
        self, chunking_mixin, sample_serialized_batch_requests
    ):
        """Test chunk_requests method with invalid strategy."""
        with pytest.raises(ValueError, match="Unknown chunking strategy"):
            chunking_mixin.chunk_requests(
                sample_serialized_batch_requests, "INVALID_STRATEGY"
            )

    def test_chunk_preserves_serialized_request_structure(
        self, chunking_mixin, sample_serialized_batch_requests
    ):
        """Test that chunking preserves the structure of SerializedBatchRequest objects."""
        chunks = chunking_mixin.chunk_by_count(sample_serialized_batch_requests, 2)

        # Verify first chunk's first request structure
        first_request = chunks[0][0]
        assert hasattr(first_request, "experiment_id")
        assert hasattr(first_request, "provider_request")
        assert isinstance(first_request.provider_request, dict)
        assert "custom_id" in first_request.provider_request
        assert "method" in first_request.provider_request
        assert "url" in first_request.provider_request
        assert "body" in first_request.provider_request

        # Verify the content is preserved
        assert first_request.provider_request["custom_id"] == "test_id_0"
        assert first_request.provider_request["method"] == "POST"
        assert first_request.provider_request["url"] == "/v1/chat/completions"

    def test_chunk_by_size_uses_size_calculation(
        self, chunking_mixin, sample_serialized_batch_requests
    ):
        """Test that chunk_by_size performs size calculation."""
        # Use a reasonable size limit that should allow all requests
        chunks = chunking_mixin.chunk_by_size(sample_serialized_batch_requests, 100)

        # Should create one chunk with all requests
        assert len(chunks) == 1
        assert len(chunks[0]) == 3

        # Use a tiny size limit that should create multiple chunks
        chunks = chunking_mixin.chunk_by_size(sample_serialized_batch_requests, 0.00001)

        # Should create multiple chunks due to size constraint
        assert len(chunks) >= 1
        all_requests = [req for chunk in chunks for req in chunk]
        assert len(all_requests) == 3

    def test_chunk_by_size_handles_single_large_request(
        self, chunking_mixin, sample_serialized_batch_requests
    ):
        """Test that chunk_by_size handles single requests larger than max size."""
        # Use extremely small max size
        max_size_mb = 0.00001  # Very small
        chunks = chunking_mixin.chunk_by_size(
            sample_serialized_batch_requests[:1], max_size_mb
        )

        # Should still create at least one chunk with the large request
        assert len(chunks) == 1
        assert len(chunks[0]) == 1

    def test_chunking_strategies_enum(self):
        """Test ChunkingStrategy enum values."""
        assert ChunkingStrategy.BY_COUNT.value == "by_count"
        assert ChunkingStrategy.BY_SIZE.value == "by_size"
        assert len(ChunkingStrategy) == 2
