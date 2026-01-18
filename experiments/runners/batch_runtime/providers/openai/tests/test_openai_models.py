"""
Tests for OpenAI Batch API response models and deserializer
"""

import pytest
from pydantic import ValidationError

from experiments.config import ExperimentId
from experiments.runners.batch_runtime.providers.openai.deserializer import \
    OpenAIBatchProviderDeserializer
from experiments.runners.batch_runtime.providers.openai.models import \
    OpenAIBatchResultLine
from experiments.runners.batch_runtime.typedefs import (BatchResult,
                                                        ExperimentFailureModes,
                                                        ProviderBatchResult)


@pytest.fixture
def openai_deserializer(mock_openai_params):
    """Create deserializer using shared mock_openai_params fixture."""
    return OpenAIBatchProviderDeserializer(mock_openai_params)


class TestOpenAIModels:
    """Test OpenAI Pydantic models"""

    def test_batch_result_line_error(self):
        """Test error OpenAIBatchResultLine model"""
        result_data = {
            "custom_id": "experiment_123",
            "response": {
                "status_code": 400,
                "request_id": "req_123",
                "body": {
                    "type": "invalid_request_error",
                    "code": "invalid_request",
                    "message": "Invalid request format",
                    "param": "messages",
                },
            },
        }

        result = OpenAIBatchResultLine(**result_data)
        assert result.custom_id == "experiment_123"
        assert result.has_error()


class TestOpenAIDeserializer:
    """Test OpenAI batch provider deserializer"""

    def test_deserialize_error(self, openai_deserializer):
        """Test deserialization with error"""
        provider_data = ProviderBatchResult(
            {
                "results": [
                    {
                        "custom_id": "experiment_123",
                        "response": {
                            "status_code": 400,
                            "request_id": "req_123",
                            "body": {
                                "type": "invalid_request_error",
                                "code": "invalid_request",
                                "message": "Invalid request format",
                                "param": "messages",
                            },
                        },
                    }
                ]
            }
        )

        result = openai_deserializer.deserialize(provider_data)
        assert result.data[0].failure_reason == ExperimentFailureModes.API_ERROR
        assert "Invalid request format" in result.data[0].response_content
