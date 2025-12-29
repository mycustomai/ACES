import os
from unittest.mock import MagicMock, patch

import pytest

from experiments.config import ExperimentId
from experiments.runners.batch_runtime.providers._base.submit import ChunkingStrategy
from experiments.runners.batch_runtime.providers.gemini.submit import (
    GeminiBatchProviderSubmitter,
)
from experiments.runners.batch_runtime.typedefs import (
    ProviderBatchRequest,
    SerializedBatchRequest,
)


@pytest.fixture
def serialized_requests():
    """Create test serialized requests."""
    requests = []
    for i in range(150):  # More than 100 to test chunking
        experiment_id = f"test_experiment_{i}"
        provider_request = ProviderBatchRequest(
            {
                "contents": [
                    {"role": "user", "parts": [{"text": f"Test message {i}"}]}
                ],
                "model": "gemini-2.0-flash-001",
                "custom_id": experiment_id,
                "generationConfig": {"maxOutputTokens": 100, "temperature": 0.7},
            }
        )
        requests.append(
            SerializedBatchRequest(
                experiment_id=ExperimentId(experiment_id),
                provider_request=provider_request,
            )
        )
    return requests


@pytest.fixture
def mock_env():
    """Set up environment variables for testing."""
    with patch.dict(
        os.environ,
        {
            "GOOGLE_API_KEY": "test-api-key",
            "GOOGLE_CLOUD_PROJECT": "test-project",
            "GCS_BUCKET_NAME": "test-bucket",
            "GOOGLE_CLOUD_LOCATION": "us-central1",
        },
    ):
        yield


class TestGeminiBatchProviderSubmitter:
    @patch("experiments.runners.batch_runtime.providers.gemini.submit.genai")
    @patch("experiments.runners.batch_runtime.providers.gemini.submit.storage")
    @patch("experiments.runners.batch_runtime.providers.gemini.submit.vertexai")
    def test_setup_success(
        self, mock_vertexai, mock_storage, mock_genai, mock_gemini_params, mock_env
    ):
        """Test successful setup with all required configuration."""
        # Mock the client creation
        mock_genai.Client.return_value = MagicMock()
        mock_storage.Client.return_value = MagicMock()

        submitter = GeminiBatchProviderSubmitter(mock_gemini_params)

        # Verify clients were initialized
        assert hasattr(submitter, "genai_client")
        assert hasattr(submitter, "storage_client")
        assert hasattr(submitter, "bucket")

        # Verify Vertex AI was initialized
        mock_vertexai.init.assert_called_once_with(
            project="test-project", location="us-central1"
        )

    def test_setup_missing_config(self, mock_gemini_params):
        """Test setup fails with missing configuration."""
        # Clear environment variables
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError) as exc_info:
                GeminiBatchProviderSubmitter(mock_gemini_params)

            assert "Missing configuration" in str(exc_info.value)

    def test_chunking_strategy(self):
        """Test the chunking strategy configuration."""
        # Verify default chunking settings on the class
        assert (
            GeminiBatchProviderSubmitter.DEFAULT_CHUNKING_STRATEGY
            == ChunkingStrategy.BY_COUNT
        )
        assert GeminiBatchProviderSubmitter.DEFAULT_CHUNK_SIZE == 1000

    def test_no_genai_client_error(self, mock_gemini_params, mock_env):
        """Test error when genai client is not configured."""
        submitter = GeminiBatchProviderSubmitter(mock_gemini_params)

        # Manually unset the client
        delattr(submitter, "genai_client")

        with pytest.raises(ValueError) as exc_info:
            submitter.submit([])

        assert "Gemini client not configured" in str(exc_info.value)
