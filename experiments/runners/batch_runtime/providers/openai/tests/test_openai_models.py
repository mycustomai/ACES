"""
Tests for OpenAI Batch API response models and deserializer
"""

import pytest
from pydantic import ValidationError

from agent.src.typedefs import EngineParams, EngineType
from experiments.config import ExperimentId
from experiments.runners.batch_runtime.providers.openai.deserializer import \
    OpenAIBatchProviderDeserializer
from experiments.runners.batch_runtime.providers.openai.models import \
    OpenAIBatchResultLine
from experiments.runners.batch_runtime.typedefs import (BatchResult,
                                                        ExperimentFailureModes,
                                                        ProviderBatchResult)


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

    def setup_method(self):
        """Set up deserializer for each test"""
        engine_params = EngineParams(
            engine_type=EngineType.OPENAI,
            model="gpt-3.5-turbo",
            temperature=0.0,
            max_new_tokens=1000,
        )
        self.deserializer = OpenAIBatchProviderDeserializer(engine_params)

    def test_deserialize_error(self):
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

        result = self.deserializer.deserialize(provider_data)
        assert result.data[0].failure_reason == ExperimentFailureModes.API_ERROR
        assert "Invalid request format" in result.data[0].response_content
