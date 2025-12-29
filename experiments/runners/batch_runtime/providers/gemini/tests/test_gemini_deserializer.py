import pytest

from agent.src.core.tools import AddToCartInput
from experiments.runners.batch_runtime.providers.gemini.deserializer import (
    GeminiBatchProviderDeserializer,
)
from experiments.runners.batch_runtime.typedefs import (
    ExperimentFailureModes,
    ProviderBatchResult,
)


@pytest.fixture
def successful_response():
    """Create a successful Gemini response with tool call."""
    return {
        "results": [
            {
                "custom_id": "test_experiment_1",
                "response": {
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    {"text": "I found the perfect mousepad for you."},
                                    {
                                        "functionCall": {
                                            "name": "add_to_cart",
                                            "args": {
                                                "product_title": "Gaming Mouse Pad XL",
                                                "price": 29.99,
                                                "rating": 4.5,
                                                "number_of_reviews": 1234,
                                            },
                                        }
                                    },
                                ]
                            }
                        }
                    ]
                },
            }
        ]
    }


@pytest.fixture
def response_without_tool_call():
    """Create a Gemini response without tool call."""
    return {
        "results": [
            {
                "custom_id": "test_experiment_2",
                "response": {
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    {"text": "I couldn't find any suitable products."}
                                ]
                            }
                        }
                    ]
                },
            }
        ]
    }


@pytest.fixture
def error_response():
    """Create an error response."""
    return {
        "results": [
            {
                "custom_id": "test_experiment_3",
                "error": {"message": "Request failed due to invalid input"},
            }
        ]
    }


@pytest.fixture
def response_no_candidates():
    """Create a response with no candidates."""
    return {
        "results": [
            {
                "custom_id": "test_experiment_4",
                "response": {"candidates": []},
            }
        ]
    }


class TestGeminiBatchProviderDeserializer:
    def test_deserialize_successful_response(self, mock_gemini_params, successful_response):
        """Test deserializing a successful response with tool call."""
        deserializer = GeminiBatchProviderDeserializer(mock_gemini_params)
        result = deserializer.deserialize(ProviderBatchResult(successful_response))

        assert len(result.data) == 1
        experiment_result = result.data[0]

        assert experiment_result.experiment_id == "test_experiment_1"
        assert (
            experiment_result.response_content
            == "I found the perfect mousepad for you."
        )
        assert experiment_result.tool_call is not None
        assert experiment_result.tool_call.product_title == "Gaming Mouse Pad XL"
        assert experiment_result.tool_call.price == 29.99
        assert experiment_result.tool_call.rating == 4.5
        assert experiment_result.tool_call.number_of_reviews == 1234
        assert experiment_result.failure_reason is None

    def test_deserialize_no_tool_call(self, mock_gemini_params, response_without_tool_call):
        """Test deserializing a response without tool call."""
        deserializer = GeminiBatchProviderDeserializer(mock_gemini_params)
        result = deserializer.deserialize(
            ProviderBatchResult(response_without_tool_call)
        )

        assert len(result.data) == 1
        experiment_result = result.data[0]

        assert experiment_result.experiment_id == "test_experiment_2"
        assert (
            experiment_result.response_content
            == "I couldn't find any suitable products."
        )
        assert experiment_result.tool_call is None
        assert experiment_result.failure_reason == ExperimentFailureModes.NO_TOOL_CALL

    def test_deserialize_error_response(self, mock_gemini_params, error_response):
        """Test deserializing an error response."""
        deserializer = GeminiBatchProviderDeserializer(mock_gemini_params)
        result = deserializer.deserialize(ProviderBatchResult(error_response))

        assert len(result.data) == 1
        experiment_result = result.data[0]

        assert experiment_result.experiment_id == "test_experiment_3"
        assert (
            experiment_result.response_content
            == "API Error: Request failed due to invalid input"
        )
        assert experiment_result.tool_call is None
        assert experiment_result.failure_reason == ExperimentFailureModes.API_ERROR

    def test_deserialize_no_candidates(self, mock_gemini_params, response_no_candidates):
        """Test deserializing a response with no candidates."""
        deserializer = GeminiBatchProviderDeserializer(mock_gemini_params)
        result = deserializer.deserialize(ProviderBatchResult(response_no_candidates))

        assert len(result.data) == 1
        experiment_result = result.data[0]

        assert experiment_result.experiment_id == "test_experiment_4"
        assert experiment_result.response_content == "No candidates in response"
        assert experiment_result.tool_call is None
        assert experiment_result.failure_reason == ExperimentFailureModes.API_ERROR

    def test_extract_text_content_multiple_parts(self, mock_gemini_params):
        """Test extracting text content from multiple text parts."""
        deserializer = GeminiBatchProviderDeserializer(mock_gemini_params)

        parts = [
            {"text": "Part 1."},
            {"text": "Part 2."},
            {"functionCall": {"name": "some_function", "args": {}}},
            {"text": "Part 3."},
        ]

        text = deserializer._extract_text_content(parts)
        assert text == "Part 1. Part 2. Part 3."

    def test_extract_text_content_no_text(self, mock_gemini_params):
        """Test extracting text when no text parts exist."""
        deserializer = GeminiBatchProviderDeserializer(mock_gemini_params)

        parts = [
            {"functionCall": {"name": "some_function", "args": {}}},
        ]

        text = deserializer._extract_text_content(parts)
        assert text == ""

    def test_extract_tool_call_valid(self, mock_gemini_params):
        """Test extracting a valid tool call."""
        deserializer = GeminiBatchProviderDeserializer(mock_gemini_params)

        parts = [
            {"text": "Here's a product:"},
            {
                "functionCall": {
                    "name": "add_to_cart",
                    "args": {
                        "product_title": "Test Product",
                        "price": 19.99,
                        "rating": 4.0,
                        "number_of_reviews": 100,
                    },
                }
            },
        ]

        tool_call = deserializer._extract_tool_call(parts)
        assert tool_call is not None
        assert isinstance(tool_call, AddToCartInput)
        assert tool_call.product_title == "Test Product"
        assert tool_call.price == 19.99
        assert tool_call.rating == 4.0
        assert tool_call.number_of_reviews == 100

    def test_extract_tool_call_invalid_args(self, mock_gemini_params):
        """Test extracting tool call with invalid arguments."""
        deserializer = GeminiBatchProviderDeserializer(mock_gemini_params)

        parts = [
            {
                "functionCall": {
                    "name": "add_to_cart",
                    "args": {
                        "product_title": "Test Product",
                        "price": "not a number",  # Invalid type
                        "rating": 4.0,
                    },
                }
            },
        ]

        tool_call = deserializer._extract_tool_call(parts)
        assert tool_call is None  # Should return None on parse error

    def test_extract_tool_call_wrong_function(self, mock_gemini_params):
        """Test extracting tool call with wrong function name."""
        deserializer = GeminiBatchProviderDeserializer(mock_gemini_params)

        parts = [
            {
                "functionCall": {
                    "name": "wrong_function",
                    "args": {"some": "data"},
                }
            },
        ]

        tool_call = deserializer._extract_tool_call(parts)
        assert tool_call is None

    def test_missing_custom_id(self, mock_gemini_params):
        """Test handling missing custom_id."""
        response = {
            "results": [
                {
                    # Missing custom_id
                    "response": {
                        "candidates": [{"content": {"parts": [{"text": "Test"}]}}]
                    },
                }
            ]
        }

        deserializer = GeminiBatchProviderDeserializer(mock_gemini_params)

        with pytest.raises(ValueError) as exc_info:
            deserializer.deserialize(ProviderBatchResult(response))

        assert "No custom_id found" in str(exc_info.value)

    def test_missing_results_key(self, mock_gemini_params):
        """Test handling missing results key."""
        response = {"data": []}  # Wrong key

        deserializer = GeminiBatchProviderDeserializer(mock_gemini_params)

        with pytest.raises(ValueError) as exc_info:
            deserializer.deserialize(ProviderBatchResult(response))

        assert "No results found" in str(exc_info.value)

    def test_multiple_results(self, mock_gemini_params):
        """Test deserializing multiple results in one batch."""
        response = {
            "results": [
                {
                    "custom_id": "exp1",
                    "response": {
                        "candidates": [{"content": {"parts": [{"text": "Response 1"}]}}]
                    },
                },
                {
                    "custom_id": "exp2",
                    "response": {
                        "candidates": [
                            {
                                "content": {
                                    "parts": [
                                        {"text": "Response 2"},
                                        {
                                            "functionCall": {
                                                "name": "add_to_cart",
                                                "args": {
                                                    "product_title": "Item",
                                                    "price": 10.0,
                                                    "rating": 5.0,
                                                    "number_of_reviews": 5,
                                                },
                                            }
                                        },
                                    ]
                                }
                            }
                        ]
                    },
                },
                {
                    "custom_id": "exp3",
                    "error": {"message": "Failed"},
                },
            ]
        }

        deserializer = GeminiBatchProviderDeserializer(mock_gemini_params)
        result = deserializer.deserialize(ProviderBatchResult(response))

        assert len(result.data) == 3

        # First result - no tool call
        assert result.data[0].experiment_id == "exp1"
        assert result.data[0].response_content == "Response 1"
        assert result.data[0].tool_call is None
        assert result.data[0].failure_reason == ExperimentFailureModes.NO_TOOL_CALL

        # Second result - with tool call
        assert result.data[1].experiment_id == "exp2"
        assert result.data[1].response_content == "Response 2"
        assert result.data[1].tool_call is not None
        assert result.data[1].failure_reason is None

        # Third result - error
        assert result.data[2].experiment_id == "exp3"
        assert result.data[2].response_content == "API Error: Failed"
        assert result.data[2].tool_call is None
        assert result.data[2].failure_reason == ExperimentFailureModes.API_ERROR

    def test_response_fallback_format(self, mock_gemini_params):
        """Test handling response in fallback format (no 'response' key)."""
        response = {
            "results": [
                {
                    "custom_id": "test_exp",
                    # Direct format without 'response' wrapper
                    "candidates": [
                        {"content": {"parts": [{"text": "Direct response format"}]}}
                    ],
                }
            ]
        }

        deserializer = GeminiBatchProviderDeserializer(mock_gemini_params)
        result = deserializer.deserialize(ProviderBatchResult(response))

        assert len(result.data) == 1
        assert result.data[0].response_content == "Direct response format"

    def test_edge_case_malformed_function_call(self, mock_gemini_params):
        """Test handling of malformed function call responses."""
        response = {
            "results": [
                {
                    "custom_id": "test_malformed",
                    "response": {
                        "candidates": [
                            {
                                "content": {
                                    "parts": [
                                        {"text": "Here's a product"},
                                        {
                                            "functionCall": {
                                                "name": "add_to_cart",
                                                "args": {
                                                    "price": "invalid_price"  # Invalid type
                                                },
                                            }
                                        },
                                    ]
                                }
                            }
                        ]
                    },
                }
            ]
        }

        deserializer = GeminiBatchProviderDeserializer(mock_gemini_params)
        result = deserializer.deserialize(ProviderBatchResult(response))

        assert len(result.data) == 1
        assert result.data[0].tool_call is None
        assert result.data[0].failure_reason == ExperimentFailureModes.NO_TOOL_CALL

    def test_edge_case_unicode_response(self, mock_gemini_params):
        """Test handling of unicode content in responses."""
        response = {
            "results": [
                {
                    "custom_id": "test_unicode",
                    "response": {
                        "candidates": [
                            {
                                "content": {
                                    "parts": [
                                        {
                                            "text": "Product with emoji 🛒 and chars: ñáéíóú"
                                        }
                                    ]
                                }
                            }
                        ]
                    },
                }
            ]
        }

        deserializer = GeminiBatchProviderDeserializer(mock_gemini_params)
        result = deserializer.deserialize(ProviderBatchResult(response))

        assert len(result.data) == 1
        assert (
            result.data[0].response_content == "Product with emoji 🛒 and chars: ñáéíóú"
        )
