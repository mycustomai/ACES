import json
from unittest.mock import Mock, patch

import pytest
from openai.types import Batch
from typing_extensions import get_args

from agent.src.typedefs import EngineParams, EngineType
from experiments.runners.batch_new.providers.openai.monitor import \
    OpenAIProviderBatchMonitor
from experiments.runners.batch_new.typedefs import (BatchStatus,
                                                    BatchStatusResult,
                                                    ProviderBatchId)


class TestOpenAIProviderBatchMonitorValidation:
    """Test validation of OpenAI status mapping."""

    def test_validate_mapping_complete_coverage(self):
        """Test that _validate_mapping passes when all statuses are covered."""
        OpenAIProviderBatchMonitor._validate_mapping()

    def test_validate_mapping_missing_status(self):
        """Test that _validate_mapping raises ValueError when status is missing."""
        incomplete_mapping = {
            "completed": BatchStatus.COMPLETED,
            "failed": BatchStatus.FAILED,
        }

        with patch.object(
            OpenAIProviderBatchMonitor, "STATUS_MAPPING", incomplete_mapping
        ):
            with pytest.raises(
                ValueError, match="Missing STATUS_MAPPING entries for statuses"
            ):
                OpenAIProviderBatchMonitor._validate_mapping()

    def test_validate_mapping_with_extra_statuses(self):
        """Test that _validate_mapping handles extra statuses gracefully."""
        original_mapping = OpenAIProviderBatchMonitor.STATUS_MAPPING
        mapping_with_extras = {
            **original_mapping,
            "unknown_status": BatchStatus.FAILED,
            "another_unknown": BatchStatus.FAILED,
        }

        with patch.object(
            OpenAIProviderBatchMonitor, "STATUS_MAPPING", mapping_with_extras
        ):
            # Should not raise exception
            OpenAIProviderBatchMonitor._validate_mapping()

    def test_validate_mapping_covers_all_openai_statuses(self):
        """Test that current STATUS_MAPPING covers all OpenAI Batch statuses."""
        allowed_statuses = set(get_args(Batch.__annotations__["status"]))
        mapped_statuses = set(OpenAIProviderBatchMonitor.STATUS_MAPPING.keys())

        assert allowed_statuses.issubset(mapped_statuses)

    def test_validate_mapping_maps_to_valid_batch_statuses(self):
        """Test that all mapped values are valid BatchStatus enum values."""
        valid_batch_statuses = set(BatchStatus)

        for (
            openai_status,
            batch_status,
        ) in OpenAIProviderBatchMonitor.STATUS_MAPPING.items():
            assert batch_status in valid_batch_statuses, (
                f"Invalid BatchStatus: {batch_status}"
            )

    def test_validate_mapping_expected_mappings(self):
        """Test that specific expected status mappings are correct."""
        mapping = OpenAIProviderBatchMonitor.STATUS_MAPPING

        assert mapping["completed"] == BatchStatus.COMPLETED
        assert mapping["failed"] == BatchStatus.FAILED
        assert mapping["expired"] == BatchStatus.FAILED
        assert mapping["cancelled"] == BatchStatus.FAILED
        assert mapping["cancelling"] == BatchStatus.FAILED
        assert mapping["in_progress"] == BatchStatus.IN_PROGRESS
        assert mapping["validating"] == BatchStatus.IN_PROGRESS
        assert mapping["finalizing"] == BatchStatus.IN_PROGRESS


class TestOpenAIProviderBatchMonitorMapping:
    """Test status mapping functionality."""

    def test_map_openai_status_known_status(self):
        """Test mapping of known OpenAI statuses."""
        assert (
            OpenAIProviderBatchMonitor._map_openai_status("completed")
            == BatchStatus.COMPLETED
        )
        assert (
            OpenAIProviderBatchMonitor._map_openai_status("failed")
            == BatchStatus.FAILED
        )
        assert (
            OpenAIProviderBatchMonitor._map_openai_status("in_progress")
            == BatchStatus.IN_PROGRESS
        )

    def test_map_openai_status_unknown_defaults_to_failed(self):
        """Test that unknown OpenAI statuses default to FAILED."""
        assert (
            OpenAIProviderBatchMonitor._map_openai_status("unknown_status")
            == BatchStatus.FAILED
        )
        assert OpenAIProviderBatchMonitor._map_openai_status("") == BatchStatus.FAILED
        assert (
            OpenAIProviderBatchMonitor._map_openai_status("nonsense")
            == BatchStatus.FAILED
        )


class TestOpenAIProviderBatchMonitorDownload:
    """Test batch result download functionality."""

    @pytest.fixture
    def mock_engine_params(self):
        """Mock engine parameters."""
        return EngineParams(
            engine_type=EngineType.OPENAI, model="gpt-4", api_key="test-key"
        )

    @pytest.fixture
    def mock_monitor(self, mock_engine_params):
        """Create a mock monitor instance."""
        with patch("openai.OpenAI"):
            monitor = OpenAIProviderBatchMonitor(mock_engine_params)
            monitor.client = Mock()
            return monitor

    def test_download_batch_results_success(self, mock_monitor):
        """Test successful batch result download."""
        batch_id = "batch_123"
        mock_batch = Mock()
        mock_batch.output_file_id = "file_456"
        mock_batch.error_file_id = None

        # Mock file content with actual OpenAI batch JSONL format
        mock_response = Mock()
        mock_response.content = b'{"id": "batch_req_1", "custom_id": "request-1", "response": {"status_code": 200, "request_id": "req_1", "body": {"choices": [{"message": {"content": "result1"}}]}}}\n{"id": "batch_req_2", "custom_id": "request-2", "response": {"status_code": 200, "request_id": "req_2", "body": {"choices": [{"message": {"content": "result2"}}]}}}'
        mock_monitor.client.files.content.return_value = mock_response

        result = mock_monitor._accumulate_batch_results(batch_id, mock_batch)

        assert isinstance(result, dict)
        assert "results" in result
        assert len(result["results"]) == 2
        assert result["results"][0]["id"] == "batch_req_1"
        assert result["results"][0]["custom_id"] == "request-1"
        assert result["results"][1]["id"] == "batch_req_2"
        assert result["results"][1]["custom_id"] == "request-2"

        mock_monitor.client.files.content.assert_called_once_with("file_456")

    def test_download_batch_results_no_output_file_id(self, mock_monitor):
        """Test download when no output file ID is present."""
        batch_id = "batch_123"
        mock_batch = Mock()
        mock_batch.output_file_id = None

        with pytest.raises(
            ValueError, match="No output file ID found for batch batch_123"
        ):
            mock_monitor._accumulate_batch_results(batch_id, mock_batch)

    def test_download_batch_results_empty_content(self, mock_monitor):
        """Test download with empty file content."""
        batch_id = "batch_123"
        mock_batch = Mock()
        mock_batch.output_file_id = "file_456"

        mock_response = Mock()
        mock_response.content = b""
        mock_monitor.client.files.content.return_value = mock_response

        result = mock_monitor._accumulate_batch_results(batch_id, mock_batch)

        assert isinstance(result, dict)
        assert "results" in result
        assert len(result["results"]) == 0

    def test_download_batch_results_single_line(self, mock_monitor):
        """Test download with single line of content."""
        batch_id = "batch_123"
        mock_batch = Mock()
        mock_batch.output_file_id = "file_456"
        mock_batch.error_file_id = None

        mock_response = Mock()
        mock_response.content = b'{"id": "batch_req_1", "custom_id": "request-1", "response": {"status_code": 200, "request_id": "req_1", "body": {"choices": [{"message": {"content": "single_result"}}]}}}'
        mock_monitor.client.files.content.return_value = mock_response

        result = mock_monitor._accumulate_batch_results(batch_id, mock_batch)

        assert isinstance(result, dict)
        assert "results" in result
        assert len(result["results"]) == 1
        assert result["results"][0]["id"] == "batch_req_1"
        assert result["results"][0]["custom_id"] == "request-1"

    def test_download_batch_results_with_empty_lines(self, mock_monitor):
        """Test download with empty lines in content."""
        batch_id = "batch_123"
        mock_batch = Mock()
        mock_batch.output_file_id = "file_456"
        mock_batch.error_file_id = None

        mock_response = Mock()
        mock_response.content = (
            b'{"id": "req_1", "response": {}}\n\n{"id": "req_2", "response": {}}\n'
        )
        mock_monitor.client.files.content.return_value = mock_response

        result = mock_monitor._accumulate_batch_results(batch_id, mock_batch)

        assert isinstance(result, dict)
        assert "results" in result
        assert len(result["results"]) == 2


class TestOpenAIProviderBatchMonitorSetup:
    """Test monitor setup functionality."""

    def test_setup_with_api_key_param(self):
        """Test setup with API key in engine params."""
        from pydantic import SecretStr

        engine_params = EngineParams(
            engine_type=EngineType.OPENAI, model="gpt-4", api_key=SecretStr("test-key")
        )

        with patch("openai.OpenAI") as mock_openai:
            monitor = OpenAIProviderBatchMonitor(engine_params)
            mock_openai.assert_called_once_with(api_key=SecretStr("test-key"))

    def test_setup_with_env_api_key(self):
        """Test setup with API key from environment."""
        engine_params = EngineParams(engine_type=EngineType.OPENAI, model="gpt-4")

        with (
            patch("openai.OpenAI") as mock_openai,
            patch.dict("os.environ", {"OPENAI_API_KEY": "env-key"}),
        ):
            monitor = OpenAIProviderBatchMonitor(engine_params)
            mock_openai.assert_called_once_with(api_key="env-key")

    def test_setup_no_api_key_raises_error(self):
        """Test setup without API key raises error."""
        engine_params = EngineParams(engine_type=EngineType.OPENAI, model="gpt-4")

        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="No OpenAI API key configured"):
                OpenAIProviderBatchMonitor(engine_params)


class TestOpenAIProviderBatchMonitorBatches:
    """Test batch monitoring functionality."""

    @pytest.fixture
    def mock_engine_params(self):
        """Mock engine parameters."""
        return EngineParams(
            engine_type=EngineType.OPENAI, model="gpt-4", api_key="test-key"
        )

    @pytest.fixture
    def mock_monitor(self, mock_engine_params):
        """Create a mock monitor instance."""
        with patch("openai.OpenAI"):
            monitor = OpenAIProviderBatchMonitor(mock_engine_params)
            monitor.client = Mock()
            return monitor

    def _create_mock_batch(self, status, output_file_id=None):
        """Helper to create mock batch with given status."""
        mock_batch = Mock()
        mock_batch.status = status
        if output_file_id:
            mock_batch.output_file_id = output_file_id
        return mock_batch

    def _setup_file_download(
        self, mock_monitor, content=b'{"id": "req_1", "response": {}}'
    ):
        """Helper to setup file download mock."""
        mock_response = Mock()
        mock_response.content = content
        mock_monitor.client.files.content.return_value = mock_response

    def _assert_single_batch_result(
        self, results, batch_id, expected_status, should_have_result=None
    ):
        """Helper to assert single batch monitoring result."""
        assert len(results) == 1
        assert isinstance(results[0], BatchStatusResult)
        assert results[0].batch_id == batch_id
        assert results[0].status == expected_status
        if should_have_result is not None:
            if should_have_result:
                assert results[0].result is not None
            else:
                assert results[0].result is None

    def test_monitor_batches_completed_batch(self, mock_monitor):
        """Test monitoring a completed batch."""
        batch_id = ProviderBatchId("batch_123")

        mock_batch = self._create_mock_batch("completed", "file_456")
        mock_monitor.client.batches.retrieve.return_value = mock_batch
        self._setup_file_download(mock_monitor)

        results = mock_monitor.monitor_batches([batch_id])

        self._assert_single_batch_result(
            results, batch_id, BatchStatus.COMPLETED, should_have_result=True
        )

    def test_monitor_batches_in_progress_batch(self, mock_monitor):
        """Test monitoring an in-progress batch."""
        batch_id = ProviderBatchId("batch_123")

        mock_batch = self._create_mock_batch("in_progress")
        mock_monitor.client.batches.retrieve.return_value = mock_batch

        results = mock_monitor.monitor_batches([batch_id])

        self._assert_single_batch_result(
            results, batch_id, BatchStatus.IN_PROGRESS, should_have_result=False
        )

    def test_monitor_batches_failed_batch(self, mock_monitor):
        """Test monitoring a failed batch."""
        batch_id = ProviderBatchId("batch_123")

        mock_batch = self._create_mock_batch("failed")
        mock_monitor.client.batches.retrieve.return_value = mock_batch

        results = mock_monitor.monitor_batches([batch_id])

        self._assert_single_batch_result(
            results, batch_id, BatchStatus.FAILED, should_have_result=False
        )

    def test_monitor_batches_multiple_batches(self, mock_monitor):
        """Test monitoring multiple batches."""
        batch_ids = [
            ProviderBatchId("batch_123"),
            ProviderBatchId("batch_456"),
            ProviderBatchId("batch_789"),
        ]

        # Mock different batch statuses
        mock_batches = [
            Mock(status="completed", output_file_id="file_1"),
            Mock(status="in_progress"),
            Mock(status="failed"),
        ]

        mock_monitor.client.batches.retrieve.side_effect = mock_batches

        # Mock file download for completed batch
        mock_response = Mock()
        mock_response.content = b'{"id": "req_1", "response": {}}'
        mock_monitor.client.files.content.return_value = mock_response

        results = mock_monitor.monitor_batches(batch_ids)

        assert len(results) == 3
        assert results[0].status == BatchStatus.COMPLETED
        assert results[0].result is not None
        assert results[1].status == BatchStatus.IN_PROGRESS
        assert results[1].result is None
        assert results[2].status == BatchStatus.FAILED
        assert results[2].result is None

    def test_monitor_batches_no_client_raises_error(self, mock_engine_params):
        """Test monitoring without client raises error."""
        with patch("openai.OpenAI"):
            monitor = OpenAIProviderBatchMonitor(mock_engine_params)
            monitor.client = None

            with pytest.raises(ValueError, match="OpenAI client not configured"):
                monitor.monitor_batches([ProviderBatchId("batch_123")])

    def test_check_batch_status(self, mock_monitor):
        """Test checking batch status."""
        batch_id = "batch_123"
        mock_batch = Mock()
        mock_monitor.client.batches.retrieve.return_value = mock_batch

        result = mock_monitor._check_batch_status(batch_id)

        assert result == mock_batch
        mock_monitor.client.batches.retrieve.assert_called_once_with(batch_id)

    def test_download_batch_results_invalid_json(self, mock_monitor):
        """Test download with invalid JSON content."""
        batch_id = "batch_123"
        mock_batch = Mock()
        mock_batch.output_file_id = "file_456"

        mock_response = Mock()
        mock_response.content = b'{"invalid": json}\n{"another": "line"}'
        mock_monitor.client.files.content.return_value = mock_response

        with pytest.raises(json.JSONDecodeError):
            mock_monitor._accumulate_batch_results(batch_id, mock_batch)

    def test_download_batch_results_with_errors(self, mock_monitor):
        """Test download with error responses in batch results."""
        batch_id = "batch_123"
        mock_batch = Mock()
        mock_batch.output_file_id = "file_456"
        mock_batch.error_file_id = None

        mock_response = Mock()
        mock_response.content = b'{"id": "batch_req_1", "custom_id": "request-1", "response": {"status_code": 400, "request_id": "req_1", "body": {"error": {"message": "Invalid request"}}}}\n{"id": "batch_req_2", "custom_id": "request-2", "response": {"status_code": 200, "request_id": "req_2", "body": {"choices": [{"message": {"content": "success"}}]}}}'
        mock_monitor.client.files.content.return_value = mock_response

        result = mock_monitor._accumulate_batch_results(batch_id, mock_batch)

        assert isinstance(result, dict)
        assert "results" in result
        assert len(result["results"]) == 2
        assert result["results"][0]["response"]["status_code"] == 400
        assert result["results"][1]["response"]["status_code"] == 200

    def _test_batch_status_mapping(
        self, mock_monitor, openai_status, expected_batch_status
    ):
        """Helper to test batch status mapping."""
        batch_id = ProviderBatchId("batch_123")

        mock_batch = self._create_mock_batch(openai_status)
        mock_monitor.client.batches.retrieve.return_value = mock_batch

        results = mock_monitor.monitor_batches([batch_id])

        self._assert_single_batch_result(
            results, batch_id, expected_batch_status, should_have_result=False
        )

    def test_monitor_batches_with_unknown_status(self, mock_monitor):
        """Test monitoring batch with unknown status defaults to FAILED."""
        self._test_batch_status_mapping(
            mock_monitor, "unknown_status", BatchStatus.FAILED
        )

    def test_monitor_batches_with_expired_status(self, mock_monitor):
        """Test monitoring batch with expired status maps to FAILED."""
        self._test_batch_status_mapping(mock_monitor, "expired", BatchStatus.FAILED)

    def test_monitor_batches_with_validating_status(self, mock_monitor):
        """Test monitoring batch with validating status maps to IN_PROGRESS."""
        self._test_batch_status_mapping(
            mock_monitor, "validating", BatchStatus.IN_PROGRESS
        )

    def test_monitor_batches_with_finalizing_status(self, mock_monitor):
        """Test monitoring batch with finalizing status maps to IN_PROGRESS."""
        self._test_batch_status_mapping(
            mock_monitor, "finalizing", BatchStatus.IN_PROGRESS
        )

    def test_monitor_batches_with_cancelled_status(self, mock_monitor):
        """Test monitoring batch with cancelled status maps to FAILED."""
        self._test_batch_status_mapping(mock_monitor, "cancelled", BatchStatus.FAILED)

    def test_monitor_batches_with_cancelling_status(self, mock_monitor):
        """Test monitoring batch with cancelling status maps to FAILED."""
        self._test_batch_status_mapping(mock_monitor, "cancelling", BatchStatus.FAILED)

    def test_monitor_batches_batch_not_found(self, mock_monitor):
        """Test monitoring when batch is not found."""
        batch_id = ProviderBatchId("batch_123")

        mock_monitor.client.batches.retrieve.return_value = None

        results = mock_monitor.monitor_batches([batch_id])

        assert len(results) == 0

    def test_monitor_batches_empty_list(self, mock_monitor):
        """Test monitoring empty batch list."""
        results = mock_monitor.monitor_batches([])

        assert len(results) == 0
        assert results == []
