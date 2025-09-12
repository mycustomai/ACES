"""
Comprehensive tests for OpenAI Batch Provider Deserializer
"""

import pytest
from pydantic import ValidationError

from agent.src.typedefs import EngineParams, EngineType
from experiments.config import ExperimentId
from experiments.runners.batch_runtime.providers.openai.deserializer import \
    OpenAIBatchProviderDeserializer
from experiments.runners.batch_runtime.typedefs import (BatchResult,
                                                        ExperimentFailureModes,
                                                        ProviderBatchResult)


class TestOpenAIBatchProviderDeserializer:
    """Comprehensive tests for OpenAI batch provider deserializer"""

    def setup_method(self):
        """Set up deserializer for each test"""
        engine_params = EngineParams(
            engine_type=EngineType.OPENAI,
            model="gpt-3.5-turbo",
            temperature=0.0,
            max_new_tokens=1000,
        )
        self.deserializer = OpenAIBatchProviderDeserializer(engine_params)

    def test_deserialize_empty_results(self):
        """Test deserializing empty results"""
        provider_data = ProviderBatchResult({"results": []})

        with pytest.raises(ValueError, match="No batch results found in the response"):
            self.deserializer.deserialize(provider_data)

    def test_deserialize_with_error_result(self):
        """Test deserializing a result with error"""
        provider_data = ProviderBatchResult(
            {
                "results": [
                    {
                        "custom_id": "exp_error",
                        "response": {
                            "status_code": 400,
                            "request_id": "req_error",
                            "body": {
                                "type": "invalid_request_error",
                                "code": "invalid_request",
                                "message": "The request was invalid",
                                "param": "messages",
                            },
                        },
                    }
                ]
            }
        )

        result = self.deserializer.deserialize(provider_data)
        assert result.data[0].failure_reason == ExperimentFailureModes.API_ERROR
        assert "The request was invalid" in result.data[0].response_content
