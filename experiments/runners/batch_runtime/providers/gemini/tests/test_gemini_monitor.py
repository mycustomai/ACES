import json
import os
from unittest.mock import MagicMock, Mock, patch

import pytest

from experiments.runners.batch_runtime.providers.gemini.monitor import (
    BatchStatus,
    GeminiProviderBatchMonitor,
    JobState,
)


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


@pytest.fixture
def mock_batch_jobs():
    """Create mock batch job objects."""
    jobs = []

    # Completed job
    job1 = Mock()
    job1.name = "batch-1"
    job1.state = JobState.JOB_STATE_SUCCEEDED
    job1.config.dest = "gs://test-bucket/batch_outputs/batch-1/"
    jobs.append(job1)

    # Running job
    job2 = Mock()
    job2.name = "batch-2"
    job2.state = JobState.JOB_STATE_RUNNING
    jobs.append(job2)

    # Failed job
    job3 = Mock()
    job3.name = "batch-3"
    job3.state = JobState.JOB_STATE_FAILED
    jobs.append(job3)

    return jobs


class TestGeminiProviderBatchMonitor:
    @patch("experiments.runners.batch_runtime.providers.gemini.monitor.genai")
    @patch("experiments.runners.batch_runtime.providers.gemini.monitor.storage")
    @patch("experiments.runners.batch_runtime.providers.gemini.monitor.vertexai")
    def test_setup_success(
        self, mock_vertexai, mock_storage, mock_genai, mock_gemini_params, mock_env
    ):
        """Test successful setup with all required configuration."""
        # Mock the client creation
        mock_genai.Client.return_value = MagicMock()
        mock_storage.Client.return_value = MagicMock()

        monitor = GeminiProviderBatchMonitor(mock_gemini_params)

        # Verify clients were initialized
        assert hasattr(monitor, "genai_client")
        assert hasattr(monitor, "storage_client")
        assert hasattr(monitor, "bucket")
        assert hasattr(monitor, "_batch_output_uri_mappings")

    def test_status_mapping(self):
        """Test JobState to BatchStatus mapping."""
        assert (
            GeminiProviderBatchMonitor.STATUS_MAPPING[JobState.JOB_STATE_SUCCEEDED]
            == BatchStatus.COMPLETED
        )
        assert (
            GeminiProviderBatchMonitor.STATUS_MAPPING[JobState.JOB_STATE_FAILED]
            == BatchStatus.FAILED
        )
        assert (
            GeminiProviderBatchMonitor.STATUS_MAPPING[JobState.JOB_STATE_CANCELLED]
            == BatchStatus.FAILED
        )
        assert (
            GeminiProviderBatchMonitor.STATUS_MAPPING[JobState.JOB_STATE_RUNNING]
            == BatchStatus.IN_PROGRESS
        )
        assert (
            GeminiProviderBatchMonitor.STATUS_MAPPING[JobState.JOB_STATE_PENDING]
            == BatchStatus.IN_PROGRESS
        )
        assert (
            GeminiProviderBatchMonitor.STATUS_MAPPING[JobState.JOB_STATE_PAUSED]
            == BatchStatus.IN_PROGRESS
        )

    @patch("experiments.runners.batch_runtime.providers.gemini.monitor.genai")
    @patch("experiments.runners.batch_runtime.providers.gemini.monitor.storage")
    @patch("experiments.runners.batch_runtime.providers.gemini.monitor.vertexai")
    def test_download_batch_results(
        self, mock_vertexai, mock_storage, mock_genai, mock_gemini_params, mock_env
    ):
        """Test downloading results from GCS."""
        # Mock clients
        mock_genai.Client.return_value = MagicMock()
        mock_storage_client = MagicMock()
        mock_storage.Client.return_value = mock_storage_client

        # Mock GCS blobs
        mock_blob1 = Mock()
        mock_blob1.size = 100
        mock_blob1.name = "predictions.jsonl"
        mock_blob1.download_as_text.return_value = "\n".join(
            [
                json.dumps(
                    {
                        "custom_id": "exp1",
                        "response": {
                            "candidates": [
                                {"content": {"parts": [{"text": "Result 1"}]}}
                            ]
                        },
                    }
                ),
                json.dumps(
                    {
                        "custom_id": "exp2",
                        "response": {
                            "candidates": [
                                {"content": {"parts": [{"text": "Result 2"}]}}
                            ]
                        },
                    }
                ),
            ]
        )

        mock_bucket = mock_storage_client.bucket.return_value
        mock_bucket.list_blobs.return_value = [mock_blob1]

        monitor = GeminiProviderBatchMonitor(mock_gemini_params)
        monitor._batch_output_uri_mappings["batch-1"] = (
            "gs://test-bucket/batch_outputs/batch-1/"
        )

        # Download results
        result = monitor._download_batch_results("batch-1")

        assert result is not None
        assert "results" in result
        assert len(result["results"]) == 2
        assert result["results"][0]["custom_id"] == "exp1"
        assert result["results"][1]["custom_id"] == "exp2"

    @patch("experiments.runners.batch_runtime.providers.gemini.monitor.genai")
    @patch("experiments.runners.batch_runtime.providers.gemini.monitor.storage")
    @patch("experiments.runners.batch_runtime.providers.gemini.monitor.vertexai")
    def test_download_no_output_location(
        self, mock_vertexai, mock_storage, mock_genai, mock_gemini_params, mock_env
    ):
        """Test handling when no output location is found."""
        # Mock clients
        mock_genai.Client.return_value = MagicMock()
        mock_storage.Client.return_value = MagicMock()

        monitor = GeminiProviderBatchMonitor(mock_gemini_params)

        # No output mapping for this batch - should raise ValueError
        with pytest.raises(ValueError, match="has no output directory URI"):
            monitor._download_batch_results("unknown-batch")

    @patch("experiments.runners.batch_runtime.providers.gemini.monitor.genai")
    @patch("experiments.runners.batch_runtime.providers.gemini.monitor.storage")
    @patch("experiments.runners.batch_runtime.providers.gemini.monitor.vertexai")
    def test_map_gemini_status(
        self, mock_vertexai, mock_storage, mock_genai, mock_gemini_params, mock_env
    ):
        """Test mapping of Gemini JobState to BatchStatus."""
        monitor = GeminiProviderBatchMonitor(mock_gemini_params)

        assert (
            monitor._map_gemini_status(JobState.JOB_STATE_SUCCEEDED)
            == BatchStatus.COMPLETED
        )
        assert (
            monitor._map_gemini_status(JobState.JOB_STATE_FAILED) == BatchStatus.FAILED
        )
        assert (
            monitor._map_gemini_status(JobState.JOB_STATE_RUNNING)
            == BatchStatus.IN_PROGRESS
        )

        # Test unknown status
        unknown_state = Mock()
        assert monitor._map_gemini_status(unknown_state) == BatchStatus.FAILED

    @patch("experiments.runners.batch_runtime.providers.gemini.monitor.genai")
    @patch("experiments.runners.batch_runtime.providers.gemini.monitor.storage")
    @patch("experiments.runners.batch_runtime.providers.gemini.monitor.vertexai")
    def test_list_recent_batches(
        self, mock_vertexai, mock_storage, mock_genai, mock_gemini_params, mock_env
    ):
        """Test listing recent batches with pagination limit."""
        # Mock clients
        mock_genai_client = MagicMock()
        mock_genai.Client.return_value = mock_genai_client
        mock_storage.Client.return_value = MagicMock()

        # Create many mock jobs
        mock_jobs = []
        for i in range(250):  # More than the 200 limit
            job = Mock()
            job.name = f"batch-{i}"
            mock_jobs.append(job)

        mock_genai_client.batches.list.return_value = iter(mock_jobs)

        monitor = GeminiProviderBatchMonitor(mock_gemini_params)
        batch_list = monitor._list_recent_batches()

        # Should stop at 200
        assert len(batch_list) == 200

    @patch("experiments.runners.batch_runtime.providers.gemini.monitor.genai")
    @patch("experiments.runners.batch_runtime.providers.gemini.monitor.storage")
    @patch("experiments.runners.batch_runtime.providers.gemini.monitor.vertexai")
    def test_gcs_uri_parsing(
        self, mock_vertexai, mock_storage, mock_genai, mock_gemini_params, mock_env
    ):
        """Test parsing of GCS URIs."""
        # Mock clients
        mock_genai.Client.return_value = MagicMock()
        mock_storage_client = MagicMock()
        mock_storage.Client.return_value = mock_storage_client

        # Mock empty blob list (no results)
        mock_storage_client.bucket.return_value.list_blobs.return_value = []

        monitor = GeminiProviderBatchMonitor(mock_gemini_params)

        # Test invalid URI format - should raise ValueError
        monitor._batch_output_uri_mappings["batch-1"] = "invalid-uri"
        with pytest.raises(ValueError, match="Invalid GCS URI format"):
            monitor._download_batch_results("batch-1")

        # Test valid URI but no files - should raise ValueError
        monitor._batch_output_uri_mappings["batch-2"] = "gs://bucket/path/"
        with pytest.raises(ValueError, match="No result files found"):
            monitor._download_batch_results("batch-2")

    def test_no_genai_client_error(self, mock_gemini_params, mock_env):
        """Test error when genai client is not configured."""
        monitor = GeminiProviderBatchMonitor(mock_gemini_params)

        # Manually unset the client
        delattr(monitor, "genai_client")

        with pytest.raises(ValueError) as exc_info:
            monitor.monitor_batches(["batch-1"])

        assert "Gemini client not configured" in str(exc_info.value)
