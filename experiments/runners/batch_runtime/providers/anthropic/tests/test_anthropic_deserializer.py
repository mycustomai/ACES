"""
Comprehensive tests for Anthropic Batch Provider Deserializer
"""

from unittest.mock import Mock

import pytest
from anthropic.types.beta import (BetaContentBlock, BetaMessage, BetaTextBlock,
                                  BetaToolUseBlock, BetaUsage)
from anthropic.types.beta.messages import (BetaMessageBatchCanceledResult,
                                           BetaMessageBatchErroredResult,
                                           BetaMessageBatchExpiredResult,
                                           BetaMessageBatchIndividualResponse,
                                           BetaMessageBatchSucceededResult)
from anthropic.types.beta_error_response import BetaErrorResponse
from pydantic import ValidationError

from agent.src.core.tools import AddToCartInput, ValidTools
from experiments.config import ExperimentId
from experiments.runners.batch_runtime.providers.anthropic.deserializer import \
    AnthropicBatchProviderDeserializer
from experiments.runners.batch_runtime.typedefs import (BatchResult,
                                                        ExperimentFailureModes,
                                                        ProviderBatchResult)


@pytest.fixture
def anthropic_deserializer(mock_anthropic_params):
    """Create deserializer using shared mock_anthropic_params fixture."""
    return AnthropicBatchProviderDeserializer(mock_anthropic_params)


class TestAnthropicBatchProviderDeserializer:
    """Comprehensive tests for Anthropic batch provider deserializer"""

    def test_deserialize_empty_results(self, anthropic_deserializer):
        """Test deserializing empty results"""
        provider_data = ProviderBatchResult({"results": []})

        result = anthropic_deserializer.deserialize(provider_data)

        assert isinstance(result, BatchResult)
        assert len(result.data) == 0

    def test_deserialize_missing_results_key(self, anthropic_deserializer):
        """Test deserializing without results key"""
        provider_data = ProviderBatchResult({"other_key": "value"})

        with pytest.raises(ValueError, match="No results found in the batch response"):
            anthropic_deserializer.deserialize(provider_data)

    def test_deserialize_successful_result_with_tool_call(self, anthropic_deserializer):
        """Test deserializing a successful result with tool call"""
        text_block = BetaTextBlock(type="text", text="I'll add this product to cart.")
        tool_block = BetaToolUseBlock(
            type="tool_use",
            id="toolu_123",
            name=ValidTools.ADD_TO_CART,
            input={
                "product_title": "Test Product",
                "price": 29.99,
                "rating": 4.5,
                "number_of_reviews": 123,
            },
        )

        message = BetaMessage(
            id="msg_123",
            type="message",
            role="assistant",
            content=[text_block, tool_block],
            model="claude-3-5-sonnet-20241022",
            stop_reason="tool_use",
            stop_sequence=None,
            usage=BetaUsage(input_tokens=10, output_tokens=15),
        )

        success_result = BetaMessageBatchSucceededResult(
            type="succeeded", message=message
        )

        individual_response = BetaMessageBatchIndividualResponse(
            custom_id="exp_123", result=success_result
        )

        provider_data = ProviderBatchResult(
            {"results": [individual_response.model_dump()]}
        )

        result = anthropic_deserializer.deserialize(provider_data)

        assert isinstance(result, BatchResult)
        assert len(result.data) == 1

        experiment_result = result.data[0]
        assert experiment_result.experiment_id == ExperimentId("exp_123")
        assert experiment_result.response_content == "I'll add this product to cart."
        assert experiment_result.tool_call is not None
        assert experiment_result.tool_call.product_title == "Test Product"
        assert experiment_result.tool_call.price == 29.99
        assert experiment_result.tool_call.rating == 4.5
        assert experiment_result.tool_call.number_of_reviews == 123
        assert experiment_result.failure_reason is None
        assert experiment_result.success is True

    def test_deserialize_successful_result_without_tool_call(self, anthropic_deserializer):
        """Test deserializing a successful result without tool call"""
        text_block = BetaTextBlock(
            type="text", text="I cannot find a suitable product."
        )

        message = BetaMessage(
            id="msg_123",
            type="message",
            role="assistant",
            content=[text_block],
            model="claude-3-5-sonnet-20241022",
            stop_reason="end_turn",
            stop_sequence=None,
            usage=BetaUsage(input_tokens=10, output_tokens=15),
        )

        success_result = BetaMessageBatchSucceededResult(
            type="succeeded", message=message
        )

        individual_response = BetaMessageBatchIndividualResponse(
            custom_id="exp_456", result=success_result
        )

        provider_data = ProviderBatchResult(
            {"results": [individual_response.model_dump()]}
        )

        result = anthropic_deserializer.deserialize(provider_data)

        assert isinstance(result, BatchResult)
        assert len(result.data) == 1

        experiment_result = result.data[0]
        assert experiment_result.experiment_id == ExperimentId("exp_456")
        assert experiment_result.response_content == "I cannot find a suitable product."
        assert experiment_result.tool_call is None
        assert experiment_result.failure_reason == ExperimentFailureModes.NO_TOOL_CALL
        assert experiment_result.success is False

    def test_deserialize_error_result(self, anthropic_deserializer):
        """Test deserializing an error result"""
        from anthropic.types.beta_error_response import BetaErrorResponse
        from anthropic.types.beta_invalid_request_error import \
            BetaInvalidRequestError

        error_result = BetaMessageBatchErroredResult(
            type="errored",
            error=BetaErrorResponse(
                type="error",
                error=BetaInvalidRequestError(
                    type="invalid_request_error", message="Request was invalid"
                ),
            ),
        )

        individual_response = BetaMessageBatchIndividualResponse(
            custom_id="exp_error", result=error_result
        )

        provider_data = ProviderBatchResult(
            {"results": [individual_response.model_dump()]}
        )

        result = anthropic_deserializer.deserialize(provider_data)

        assert isinstance(result, BatchResult)
        assert len(result.data) == 1

        experiment_result = result.data[0]
        assert experiment_result.experiment_id == ExperimentId("exp_error")
        assert experiment_result.response_content == "API Error: Result is errored"
        assert experiment_result.tool_call is None
        assert experiment_result.failure_reason == ExperimentFailureModes.API_ERROR
        assert experiment_result.success is False

    def test_deserialize_canceled_result(self, anthropic_deserializer):
        """Test deserializing a canceled result"""
        canceled_result = BetaMessageBatchCanceledResult(type="canceled")

        individual_response = BetaMessageBatchIndividualResponse(
            custom_id="exp_canceled", result=canceled_result
        )

        provider_data = ProviderBatchResult(
            {"results": [individual_response.model_dump()]}
        )

        result = anthropic_deserializer.deserialize(provider_data)

        assert isinstance(result, BatchResult)
        assert len(result.data) == 1

        experiment_result = result.data[0]
        assert experiment_result.experiment_id == ExperimentId("exp_canceled")
        assert experiment_result.response_content == "API Error: Result is canceled"
        assert experiment_result.tool_call is None
        assert experiment_result.failure_reason == ExperimentFailureModes.API_ERROR
        assert experiment_result.success is False

    def test_deserialize_expired_result(self, anthropic_deserializer):
        """Test deserializing an expired result"""
        expired_result = BetaMessageBatchExpiredResult(type="expired")

        individual_response = BetaMessageBatchIndividualResponse(
            custom_id="exp_expired", result=expired_result
        )

        provider_data = ProviderBatchResult(
            {"results": [individual_response.model_dump()]}
        )

        result = anthropic_deserializer.deserialize(provider_data)

        assert isinstance(result, BatchResult)
        assert len(result.data) == 1

        experiment_result = result.data[0]
        assert experiment_result.experiment_id == ExperimentId("exp_expired")
        assert experiment_result.response_content == "API Error: Result is expired"
        assert experiment_result.tool_call is None
        assert experiment_result.failure_reason == ExperimentFailureModes.API_ERROR
        assert experiment_result.success is False

    def test_deserialize_multiple_results(self, anthropic_deserializer):
        """Test deserializing multiple results"""
        # Success result
        text_block1 = BetaTextBlock(type="text", text="Added to cart.")
        tool_block1 = BetaToolUseBlock(
            type="tool_use",
            id="toolu_123",
            name=ValidTools.ADD_TO_CART,
            input={
                "product_title": "Product 1",
                "price": 19.99,
                "rating": 4.0,
                "number_of_reviews": 50,
            },
        )

        message1 = BetaMessage(
            id="msg_1",
            type="message",
            role="assistant",
            content=[text_block1, tool_block1],
            model="claude-3-5-sonnet-20241022",
            stop_reason="tool_use",
            stop_sequence=None,
            usage=BetaUsage(input_tokens=10, output_tokens=15),
        )

        success_result = BetaMessageBatchSucceededResult(
            type="succeeded", message=message1
        )

        # Error result
        from anthropic.types.beta_error_response import BetaErrorResponse
        from anthropic.types.beta_rate_limit_error import BetaRateLimitError

        error_result = BetaMessageBatchErroredResult(
            type="errored",
            error=BetaErrorResponse(
                type="error",
                error=BetaRateLimitError(
                    type="rate_limit_error", message="Rate limit exceeded"
                ),
            ),
        )

        individual_response1 = BetaMessageBatchIndividualResponse(
            custom_id="exp_success", result=success_result
        )

        individual_response2 = BetaMessageBatchIndividualResponse(
            custom_id="exp_error", result=error_result
        )

        provider_data = ProviderBatchResult(
            {
                "results": [
                    individual_response1.model_dump(),
                    individual_response2.model_dump(),
                ]
            }
        )

        result = anthropic_deserializer.deserialize(provider_data)

        assert isinstance(result, BatchResult)
        assert len(result.data) == 2

        # Check success result
        success_exp = result.data[0]
        assert success_exp.experiment_id == ExperimentId("exp_success")
        assert success_exp.success is True
        assert success_exp.tool_call is not None

        # Check error result
        error_exp = result.data[1]
        assert error_exp.experiment_id == ExperimentId("exp_error")
        assert error_exp.success is False
        assert error_exp.failure_reason == ExperimentFailureModes.API_ERROR

    def test_deserialize_unknown_result_type(self, anthropic_deserializer):
        """Test deserializing with unknown result type"""
        # Create invalid result data that will fail validation
        invalid_result_data = {
            "custom_id": "exp_unknown",
            "result": {"type": "unknown_type"},  # Invalid type
        }

        provider_data = ProviderBatchResult({"results": [invalid_result_data]})

        with pytest.raises((ValueError, ValidationError)):
            anthropic_deserializer.deserialize(provider_data)


class TestAnthropicDeserializerTextExtraction:
    """Test text content extraction functionality"""

    def test_extract_text_content_single_text_block(self):
        """Test extracting text from single text block"""
        text_block = BetaTextBlock(type="text", text="This is a response.")
        content = [text_block]

        result = AnthropicBatchProviderDeserializer._extract_text_content(content)

        assert result == "This is a response."

    def test_extract_text_content_no_text_blocks(self):
        """Test extracting text when no text blocks exist"""
        tool_block = BetaToolUseBlock(
            type="tool_use", id="toolu_123", name="test_tool", input={}
        )
        content = [tool_block]

        with pytest.raises(ValueError, match="No text blocks found in the response"):
            AnthropicBatchProviderDeserializer._extract_text_content(content)

    def test_extract_text_content_multiple_text_blocks(self):
        """Test extracting text when multiple text blocks exist"""
        text_block1 = BetaTextBlock(type="text", text="First text.")
        text_block2 = BetaTextBlock(type="text", text="Second text.")
        content = [text_block1, text_block2]

        with pytest.raises(
            ValueError, match="Multiple text blocks found in the response"
        ):
            AnthropicBatchProviderDeserializer._extract_text_content(content)

    def test_extract_text_content_empty_content(self):
        """Test extracting text from empty content"""
        content = []

        with pytest.raises(ValueError, match="No text blocks found in the response"):
            AnthropicBatchProviderDeserializer._extract_text_content(content)

    def test_extract_text_content_mixed_content(self):
        """Test extracting text from mixed content with text and tool blocks"""
        text_block = BetaTextBlock(type="text", text="I'll add this to cart.")
        tool_block = BetaToolUseBlock(
            type="tool_use",
            id="toolu_123",
            name=ValidTools.ADD_TO_CART,
            input={
                "product_title": "Test",
                "price": 10.0,
                "rating": 5.0,
                "number_of_reviews": 1,
            },
        )
        content = [text_block, tool_block]

        result = AnthropicBatchProviderDeserializer._extract_text_content(content)

        assert result == "I'll add this to cart."


class TestAnthropicDeserializerToolExtraction:
    """Test tool call extraction functionality"""

    def test_extract_tool_call_valid_add_to_cart(self):
        """Test extracting valid add_to_cart tool call"""
        tool_block = BetaToolUseBlock(
            type="tool_use",
            id="toolu_123",
            name=ValidTools.ADD_TO_CART,
            input={
                "product_title": "Test Product",
                "price": 29.99,
                "rating": 4.5,
                "number_of_reviews": 123,
            },
        )
        content = [tool_block]

        result = AnthropicBatchProviderDeserializer._extract_tool_call(content)

        assert result is not None
        assert isinstance(result, AddToCartInput)
        assert result.product_title == "Test Product"
        assert result.price == 29.99
        assert result.rating == 4.5
        assert result.number_of_reviews == 123

    def test_extract_tool_call_no_tool_blocks(self):
        """Test extracting tool call when no tool blocks exist"""
        text_block = BetaTextBlock(type="text", text="Just text.")
        content = [text_block]

        result = AnthropicBatchProviderDeserializer._extract_tool_call(content)

        assert result is None

    def test_extract_tool_call_empty_content(self):
        """Test extracting tool call from empty content"""
        content = []

        result = AnthropicBatchProviderDeserializer._extract_tool_call(content)

        assert result is None

    def test_extract_tool_call_multiple_tool_blocks(self):
        """Test extracting tool call when multiple tool blocks exist"""
        tool_block1 = BetaToolUseBlock(
            type="tool_use",
            id="toolu_123",
            name=ValidTools.ADD_TO_CART,
            input={
                "product_title": "Product 1",
                "price": 10.0,
                "rating": 5.0,
                "number_of_reviews": 1,
            },
        )
        tool_block2 = BetaToolUseBlock(
            type="tool_use",
            id="toolu_456",
            name=ValidTools.ADD_TO_CART,
            input={
                "product_title": "Product 2",
                "price": 20.0,
                "rating": 4.0,
                "number_of_reviews": 2,
            },
        )
        content = [tool_block1, tool_block2]

        with pytest.raises(
            ValueError, match="Multiple tool calls found in the response"
        ):
            AnthropicBatchProviderDeserializer._extract_tool_call(content)

    def test_extract_tool_call_non_add_to_cart_tool(self):
        """Test extracting tool call for non-add_to_cart tool"""
        tool_block = BetaToolUseBlock(
            type="tool_use", id="toolu_123", name="other_tool", input={"param": "value"}
        )
        content = [tool_block]

        result = AnthropicBatchProviderDeserializer._extract_tool_call(content)

        assert result is None

    def test_extract_tool_call_invalid_tool_input(self):
        """Test extracting tool call with invalid input"""
        tool_block = BetaToolUseBlock(
            type="tool_use",
            id="toolu_123",
            name=ValidTools.ADD_TO_CART,
            input={
                "product_title": "Test Product",
                "price": "invalid_price",  # Should be float
                "rating": 4.5,
                "number_of_reviews": 123,
            },
        )
        content = [tool_block]

        with pytest.raises(ValueError, match="Failed to parse tool input"):
            AnthropicBatchProviderDeserializer._extract_tool_call(content)

    def test_extract_tool_call_missing_required_fields(self):
        """Test extracting tool call with missing required fields"""
        tool_block = BetaToolUseBlock(
            type="tool_use",
            id="toolu_123",
            name=ValidTools.ADD_TO_CART,
            input={
                "product_title": "Test Product",
                # Missing price, rating, number_of_reviews
            },
        )
        content = [tool_block]

        with pytest.raises(ValueError, match="Failed to parse tool input"):
            AnthropicBatchProviderDeserializer._extract_tool_call(content)

    def test_extract_tool_call_mixed_content(self):
        """Test extracting tool call from mixed content"""
        text_block = BetaTextBlock(type="text", text="I'll add this to cart.")
        tool_block = BetaToolUseBlock(
            type="tool_use",
            id="toolu_123",
            name=ValidTools.ADD_TO_CART,
            input={
                "product_title": "Test Product",
                "price": 15.99,
                "rating": 3.5,
                "number_of_reviews": 45,
            },
        )
        content = [text_block, tool_block]

        result = AnthropicBatchProviderDeserializer._extract_tool_call(content)

        assert result is not None
        assert isinstance(result, AddToCartInput)
        assert result.product_title == "Test Product"
        assert result.price == 15.99
        assert result.rating == 3.5
        assert result.number_of_reviews == 45
