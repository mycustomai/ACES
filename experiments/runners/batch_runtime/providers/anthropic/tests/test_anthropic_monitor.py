"""
Comprehensive tests for Anthropic Batch Provider Monitor
"""

import os
from unittest.mock import MagicMock, Mock, patch

import pytest
from anthropic.types.beta.messages import (BetaMessageBatch,
                                           BetaMessageBatchIndividualResponse)
from typing_extensions import get_args

from agent.src.typedefs import EngineParams, EngineType
from experiments.runners.batch_runtime.providers.anthropic.monitor import \
    AnthropicBatchProviderMonitor
from experiments.runners.batch_runtime.typedefs import (BatchStatus,
                                                        BatchStatusResult,
                                                        ProviderBatchId,
                                                        ProviderBatchResult)


class TestAnthropicBatchProviderMonitorValidation:
    """Test validation of Anthropic status mapping."""

    def test_validate_mapping_complete_coverage(self):
        """Test that _validate_mapping passes when all statuses are covered."""
        AnthropicBatchProviderMonitor._validate_mapping()

    def test_validate_mapping_missing_status(self):
        """Test that _validate_mapping raises ValueError when status is missing."""
        incomplete_mapping = {
            "ended": BatchStatus.COMPLETED,
            "in_progress": BatchStatus.IN_PROGRESS,
        }

        with patch.object(
            AnthropicBatchProviderMonitor, "STATUS_MAPPING", incomplete_mapping
        ):
            with pytest.raises(
                ValueError, match="Missing STATUS_MAPPING entries for statuses"
            ):
                AnthropicBatchProviderMonitor._validate_mapping()

    def test_validate_mapping_with_extra_statuses(self):
        """Test that _validate_mapping handles extra statuses gracefully."""
        original_mapping = AnthropicBatchProviderMonitor.STATUS_MAPPING
        mapping_with_extras = {
            **original_mapping,
            "unknown_status": BatchStatus.FAILED,
            "another_unknown": BatchStatus.FAILED,
        }

        with patch.object(
            AnthropicBatchProviderMonitor, "STATUS_MAPPING", mapping_with_extras
        ):
            # Should not raise exception
            AnthropicBatchProviderMonitor._validate_mapping()

    def test_validate_mapping_covers_all_anthropic_statuses(self):
        """Test that current STATUS_MAPPING covers all Anthropic batch statuses."""
        allowed_statuses = set(
            get_args(BetaMessageBatch.__annotations__["processing_status"])
        )
        mapped_statuses = set(AnthropicBatchProviderMonitor.STATUS_MAPPING.keys())

        assert allowed_statuses.issubset(mapped_statuses)

    def test_validate_mapping_maps_to_valid_batch_statuses(self):
        """Test that all mapped values are valid BatchStatus enum values."""
        valid_batch_statuses = set(BatchStatus)

        for (
            anthropic_status,
            batch_status,
        ) in AnthropicBatchProviderMonitor.STATUS_MAPPING.items():
            assert batch_status in valid_batch_statuses, (
                f"Invalid BatchStatus: {batch_status}"
            )

    def test_validate_mapping_expected_mappings(self):
        """Test that specific expected status mappings are correct."""
        mapping = AnthropicBatchProviderMonitor.STATUS_MAPPING

        assert mapping["ended"] == BatchStatus.COMPLETED
        assert mapping["in_progress"] == BatchStatus.IN_PROGRESS
        assert mapping["canceling"] == BatchStatus.FAILED


class TestAnthropicBatchProviderMonitorMapping:
    """Test status mapping functionality."""

    def test_map_anthropic_status_known_status(self):
        """Test mapping of known Anthropic statuses."""
        assert (
            AnthropicBatchProviderMonitor._map_anthropic_status("ended")
            == BatchStatus.COMPLETED
        )
        assert (
            AnthropicBatchProviderMonitor._map_anthropic_status("in_progress")
            == BatchStatus.IN_PROGRESS
        )
        assert (
            AnthropicBatchProviderMonitor._map_anthropic_status("canceling")
            == BatchStatus.FAILED
        )

    def test_map_anthropic_status_unknown_defaults_to_failed(self):
        """Test that unknown Anthropic statuses default to FAILED."""
        assert (
            AnthropicBatchProviderMonitor._map_anthropic_status("unknown_status")
            == BatchStatus.FAILED
        )
        assert (
            AnthropicBatchProviderMonitor._map_anthropic_status("")
            == BatchStatus.FAILED
        )
        assert (
            AnthropicBatchProviderMonitor._map_anthropic_status("nonsense")
            == BatchStatus.FAILED
        )


class TestAnthropicBatchProviderMonitorSetup:
    """Test monitor setup functionality."""

    def test_setup_with_api_key_param(self):
        """Test setup with API key in engine params."""
        from pydantic import SecretStr

        engine_params = EngineParams(
            engine_type=EngineType.ANTHROPIC,
            model="claude-3-5-sonnet-20241022",
            display_name="Claude 3.5 Sonnet",
            api_key=SecretStr("test-key"),
        )

        with patch("anthropic.Anthropic") as mock_anthropic:
            monitor = AnthropicBatchProviderMonitor(engine_params)
            mock_anthropic.assert_called_once_with(api_key=SecretStr("test-key"))

    def test_setup_with_env_api_key(self):
        """Test setup with API key from environment."""
        engine_params = EngineParams(
            engine_type=EngineType.ANTHROPIC,
            model="claude-3-5-sonnet-20241022",
            display_name="Claude 3.5 Sonnet",
        )

        with (
            patch("anthropic.Anthropic") as mock_anthropic,
            patch.dict("os.environ", {"ANTHROPIC_API_KEY": "env-key"}),
        ):
            monitor = AnthropicBatchProviderMonitor(engine_params)
            mock_anthropic.assert_called_once_with(api_key="env-key")

    def test_setup_no_api_key_raises_error(self):
        """Test setup without API key raises error."""
        engine_params = EngineParams(
            engine_type=EngineType.ANTHROPIC,
            model="claude-3-5-sonnet-20241022",
            display_name="Claude 3.5 Sonnet",
        )

        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="No Anthropic API key configured"):
                AnthropicBatchProviderMonitor(engine_params)

    def test_setup_validates_mapping(self):
        """Test that setup validates the status mapping."""
        engine_params = EngineParams(
            engine_type=EngineType.ANTHROPIC,
            model="claude-3-5-sonnet-20241022",
            display_name="Claude 3.5 Sonnet",
            api_key="test-key",
        )

        with (
            patch("anthropic.Anthropic"),
            patch.object(
                AnthropicBatchProviderMonitor, "_validate_mapping"
            ) as mock_validate,
        ):
            AnthropicBatchProviderMonitor(engine_params)
            mock_validate.assert_called_once()


class TestAnthropicBatchProviderMonitorBatches:
    """Test batch monitoring functionality."""

    @pytest.fixture
    def mock_monitor(self, mock_anthropic_params):
        """Create a mock monitor instance."""
        with patch("anthropic.Anthropic"):
            monitor = AnthropicBatchProviderMonitor(mock_anthropic_params)
            monitor.client = Mock()
            return monitor

    def _create_mock_batch(self, status):
        """Helper to create mock batch with given status."""
        mock_batch = Mock(spec=BetaMessageBatch)
        mock_batch.processing_status = status
        return mock_batch

    def _create_mock_individual_response(self, custom_id="test_id"):
        """Helper to create mock individual response."""
        mock_response = Mock(spec=BetaMessageBatchIndividualResponse)
        mock_response.custom_id = custom_id
        mock_response.model_dump.return_value = {
            "custom_id": custom_id,
            "result": {"type": "succeeded"},
        }
        return mock_response

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

        mock_batch = self._create_mock_batch("ended")
        mock_monitor.client.beta.messages.batches.retrieve.return_value = mock_batch

        # Mock batch results
        mock_results = [self._create_mock_individual_response("exp_1")]
        mock_monitor.client.beta.messages.batches.results.return_value = mock_results

        results = mock_monitor.monitor_batches([batch_id])

        self._assert_single_batch_result(
            results, batch_id, BatchStatus.COMPLETED, should_have_result=True
        )
        assert results[0].result["results"] == [
            {"custom_id": "exp_1", "result": {"type": "succeeded"}}
        ]

    def test_monitor_batches_in_progress_batch(self, mock_monitor):
        """Test monitoring an in-progress batch."""
        batch_id = ProviderBatchId("batch_123")

        mock_batch = self._create_mock_batch("in_progress")
        mock_monitor.client.beta.messages.batches.retrieve.return_value = mock_batch

        results = mock_monitor.monitor_batches([batch_id])

        self._assert_single_batch_result(
            results, batch_id, BatchStatus.IN_PROGRESS, should_have_result=False
        )

    def test_monitor_batches_failed_batch(self, mock_monitor):
        """Test monitoring a failed batch."""
        batch_id = ProviderBatchId("batch_123")

        mock_batch = self._create_mock_batch("canceling")
        mock_monitor.client.beta.messages.batches.retrieve.return_value = mock_batch

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
            self._create_mock_batch("ended"),
            self._create_mock_batch("in_progress"),
            self._create_mock_batch("canceling"),
        ]

        mock_monitor.client.beta.messages.batches.retrieve.side_effect = mock_batches

        # Mock batch results for completed batch
        mock_results = [self._create_mock_individual_response("exp_1")]
        mock_monitor.client.beta.messages.batches.results.return_value = mock_results

        results = mock_monitor.monitor_batches(batch_ids)

        assert len(results) == 3
        assert results[0].status == BatchStatus.COMPLETED
        assert results[0].result is not None
        assert results[1].status == BatchStatus.IN_PROGRESS
        assert results[1].result is None
        assert results[2].status == BatchStatus.FAILED
        assert results[2].result is None

    def test_monitor_batches_no_client_raises_error(self, mock_anthropic_params):
        """Test monitoring without client raises error."""
        with patch("anthropic.Anthropic"):
            monitor = AnthropicBatchProviderMonitor(mock_anthropic_params)
            monitor.client = None

            with pytest.raises(ValueError, match="Anthropic client not configured"):
                monitor.monitor_batches([ProviderBatchId("batch_123")])

    def test_monitor_batches_empty_list(self, mock_monitor):
        """Test monitoring empty batch list."""
        results = mock_monitor.monitor_batches([])

        assert len(results) == 0
        assert results == []

    def test_monitor_batches_batch_not_found(self, mock_monitor):
        """Test monitoring when batch is not found."""
        batch_id = ProviderBatchId("batch_123")

        mock_monitor.client.beta.messages.batches.retrieve.return_value = None

        with pytest.raises(ValueError, match="Batch batch_123 not found"):
            mock_monitor.monitor_batches([batch_id])

    def test_monitor_batches_with_unknown_status(self, mock_monitor):
        """Test monitoring batch with unknown status defaults to FAILED."""
        batch_id = ProviderBatchId("batch_123")

        mock_batch = self._create_mock_batch("unknown_status")
        mock_monitor.client.beta.messages.batches.retrieve.return_value = mock_batch

        results = mock_monitor.monitor_batches([batch_id])

        self._assert_single_batch_result(
            results, batch_id, BatchStatus.FAILED, should_have_result=False
        )


class TestAnthropicBatchProviderMonitorResultsAccumulation:
    """Test batch result accumulation functionality."""

    @pytest.fixture
    def mock_monitor(self, mock_anthropic_params):
        """Create a mock monitor instance."""
        with patch("anthropic.Anthropic"):
            monitor = AnthropicBatchProviderMonitor(mock_anthropic_params)
            monitor.client = Mock()
            return monitor

    def test_accumulate_batch_results_success(self, mock_monitor):
        """Test successful batch result accumulation."""
        batch_id = "batch_123"

        # Mock individual responses
        mock_response1 = Mock(spec=BetaMessageBatchIndividualResponse)
        mock_response1.custom_id = "exp_1"
        mock_response1.model_dump.return_value = {
            "custom_id": "exp_1",
            "result": {"type": "succeeded"},
        }

        mock_response2 = Mock(spec=BetaMessageBatchIndividualResponse)
        mock_response2.custom_id = "exp_2"
        mock_response2.model_dump.return_value = {
            "custom_id": "exp_2",
            "result": {"type": "errored"},
        }

        mock_results = [mock_response1, mock_response2]
        mock_monitor.client.beta.messages.batches.results.return_value = mock_results

        result = mock_monitor._accumulate_batch_results(batch_id)

        assert isinstance(result, dict)
        assert "results" in result
        assert len(result["results"]) == 2
        assert result["results"][0]["custom_id"] == "exp_1"
        assert result["results"][1]["custom_id"] == "exp_2"

    def test_accumulate_batch_results_api_error(self, mock_monitor):
        """Test batch result accumulation with API error."""
        batch_id = "batch_123"

        mock_monitor.client.beta.messages.batches.results.side_effect = Exception(
            "API Error"
        )

        with pytest.raises(Exception, match="API Error"):
            mock_monitor._accumulate_batch_results(batch_id)

    def test_accumulate_batch_results_empty_results(self, mock_monitor):
        """Test batch result accumulation with empty results."""
        batch_id = "batch_123"

        mock_monitor.client.beta.messages.batches.results.return_value = []

        result = mock_monitor._accumulate_batch_results(batch_id)

        assert isinstance(result, dict)
        assert "results" in result
        assert len(result["results"]) == 0

    def test_accumulate_batch_results_processing_error(self, mock_monitor):
        """Test batch result accumulation with processing error."""
        batch_id = "batch_123"

        # Mock result that raises exception during processing
        mock_response = Mock()
        mock_response.model_dump.side_effect = Exception("Processing error")

        mock_monitor.client.beta.messages.batches.results.return_value = [mock_response]

        with pytest.raises(Exception, match="Processing error"):
            mock_monitor._accumulate_batch_results(batch_id)

    def test_process_batch_results_success(self):
        """Test successful batch result processing."""
        mock_response1 = Mock(spec=BetaMessageBatchIndividualResponse)
        mock_response1.model_dump.return_value = {
            "custom_id": "exp_1",
            "result": {"type": "succeeded"},
        }

        mock_response2 = Mock(spec=BetaMessageBatchIndividualResponse)
        mock_response2.model_dump.return_value = {
            "custom_id": "exp_2",
            "result": {"type": "errored"},
        }

        results_generator = [mock_response1, mock_response2]

        processed = list(
            AnthropicBatchProviderMonitor._process_batch_results(results_generator)
        )

        assert len(processed) == 2
        assert processed[0]["custom_id"] == "exp_1"
        assert processed[1]["custom_id"] == "exp_2"

    def test_process_batch_results_non_pydantic_type(self):
        """Test batch result processing with non-pydantic type."""
        # Mock non-pydantic result
        mock_response = Mock()
        del mock_response.model_dump  # Remove the model_dump attribute

        results_generator = [mock_response]

        with pytest.raises(
            ValueError, match="Non-pydantic type received as batch result line"
        ):
            list(
                AnthropicBatchProviderMonitor._process_batch_results(results_generator)
            )

    def test_process_batch_results_empty_generator(self):
        """Test batch result processing with empty generator."""
        results_generator = []

        processed = list(
            AnthropicBatchProviderMonitor._process_batch_results(results_generator)
        )

        assert len(processed) == 0

    def test_process_batch_results_model_dump_error(self):
        """Test batch result processing with model_dump error."""
        mock_response = Mock(spec=BetaMessageBatchIndividualResponse)
        mock_response.model_dump.side_effect = Exception("Model dump error")

        results_generator = [mock_response]

        with pytest.raises(Exception, match="Model dump error"):
            list(
                AnthropicBatchProviderMonitor._process_batch_results(results_generator)
            )


class TestAnthropicBatchProviderMonitorIntegration:
    """Integration tests for complete monitoring workflow."""

    @pytest.fixture
    def mock_monitor(self, mock_anthropic_params):
        """Create a mock monitor instance."""
        with patch("anthropic.Anthropic"):
            monitor = AnthropicBatchProviderMonitor(mock_anthropic_params)
            monitor.client = Mock()
            return monitor

    def test_full_monitoring_workflow_completed_batch(self, mock_monitor):
        """Test complete monitoring workflow for a completed batch."""
        batch_id = ProviderBatchId("batch_123")

        # Setup batch status
        mock_batch = Mock(spec=BetaMessageBatch)
        mock_batch.processing_status = "ended"
        mock_monitor.client.beta.messages.batches.retrieve.return_value = mock_batch

        # Setup batch results
        mock_response = Mock(spec=BetaMessageBatchIndividualResponse)
        mock_response.custom_id = "exp_1"
        mock_response.model_dump.return_value = {
            "custom_id": "exp_1",
            "result": {
                "type": "succeeded",
                "message": {
                    "id": "msg_1",
                    "content": [{"type": "text", "text": "Response text"}],
                },
            },
        }

        mock_monitor.client.beta.messages.batches.results.return_value = [mock_response]

        # Monitor the batch
        results = mock_monitor.monitor_batches([batch_id])

        # Verify results
        assert len(results) == 1
        result = results[0]
        assert result.batch_id == batch_id
        assert result.status == BatchStatus.COMPLETED
        assert result.result is not None
        assert "results" in result.result
        assert len(result.result["results"]) == 1
        assert result.result["results"][0]["custom_id"] == "exp_1"

    def test_full_monitoring_workflow_multiple_batches_mixed_statuses(
        self, mock_monitor
    ):
        """Test complete monitoring workflow for multiple batches with mixed statuses."""
        batch_ids = [
            ProviderBatchId("batch_completed"),
            ProviderBatchId("batch_in_progress"),
            ProviderBatchId("batch_failed"),
        ]

        # Setup batch statuses
        mock_batches = [
            Mock(spec=BetaMessageBatch, processing_status="ended"),
            Mock(spec=BetaMessageBatch, processing_status="in_progress"),
            Mock(spec=BetaMessageBatch, processing_status="canceling"),
        ]
        mock_monitor.client.beta.messages.batches.retrieve.side_effect = mock_batches

        # Setup batch results for completed batch only
        mock_response = Mock(spec=BetaMessageBatchIndividualResponse)
        mock_response.custom_id = "exp_completed"
        mock_response.model_dump.return_value = {
            "custom_id": "exp_completed",
            "result": {"type": "succeeded"},
        }
        mock_monitor.client.beta.messages.batches.results.return_value = [mock_response]

        # Monitor the batches
        results = mock_monitor.monitor_batches(batch_ids)

        # Verify results
        assert len(results) == 3

        # Completed batch
        assert results[0].batch_id == batch_ids[0]
        assert results[0].status == BatchStatus.COMPLETED
        assert results[0].result is not None

        # In-progress batch
        assert results[1].batch_id == batch_ids[1]
        assert results[1].status == BatchStatus.IN_PROGRESS
        assert results[1].result is None

        # Failed batch
        assert results[2].batch_id == batch_ids[2]
        assert results[2].status == BatchStatus.FAILED
        assert results[2].result is None

    def test_monitoring_with_api_failures(self, mock_monitor):
        """Test monitoring resilience with API failures."""
        batch_id = ProviderBatchId("batch_123")

        # Simulate API failure during batch status retrieval
        mock_monitor.client.beta.messages.batches.retrieve.side_effect = Exception(
            "API failure"
        )

        with pytest.raises(Exception, match="API failure"):
            mock_monitor.monitor_batches([batch_id])

    def test_monitoring_with_result_download_failures(self, mock_monitor):
        """Test monitoring with result download failures."""
        batch_id = ProviderBatchId("batch_123")

        # Setup completed batch
        mock_batch = Mock(spec=BetaMessageBatch)
        mock_batch.processing_status = "ended"
        mock_monitor.client.beta.messages.batches.retrieve.return_value = mock_batch

        # Simulate failure during result download
        mock_monitor.client.beta.messages.batches.results.side_effect = Exception(
            "Download failure"
        )

        with pytest.raises(Exception, match="Download failure"):
            mock_monitor.monitor_batches([batch_id])
