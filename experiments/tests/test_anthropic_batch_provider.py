#!/usr/bin/env python3
"""
Comprehensive pytest suite for Anthropic Batch API implementation.
Tests ensure compliance with Anthropic API documentation and proper functioning
of all batch provider methods.
"""

import json
import pytest
from pathlib import Path
from typing import Dict, List, Any
from unittest.mock import Mock, patch
import pandas as pd

# Import the modules we're testing
from experiments.runners.batch_runtime.providers.anthropic import AnthropicBatchProvider
from experiments.runners.batch_runtime.services.file_operations import FileOperationsService
from agent.src.typedefs import EngineParams, EngineType
from agent.src.core.tools import AddToCartInput


class MockExperimentData:
    """Mock ExperimentData for testing."""
    
    def __init__(self, query: str = "mousepad", experiment_label: str = "baseline", 
                 experiment_number: int = 1):
        self.query = query
        self.experiment_label = experiment_label
        self.experiment_number = experiment_number
        self.experiment_df = pd.DataFrame({
            'product_title': ['SteelSeries QcK Gaming Mouse Pad', 'Corsair MM300'],
            'price': [12.99, 24.99],
            'rating': [4.2, 4.8]
        })


@pytest.fixture
def mock_file_ops():
    """Create a mock FileOperationsService for testing."""
    mock_service = Mock(spec=FileOperationsService)
    mock_service.output_dir = Path("/tmp/test_output")
    mock_service.directories = Mock()
    mock_service.directories.batch_metadata = Path("/tmp/test_output/batch_metadata")
    mock_service.directories.batch_results = Path("/tmp/test_output/batch_results")
    mock_service.directories.batch_inputs = Path("/tmp/test_output/batch_inputs")
    return mock_service


@pytest.fixture
def anthropic_provider(mock_file_ops):
    """Create an AnthropicBatchProvider instance for testing."""
    with patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test_api_key'}):
        return AnthropicBatchProvider(mock_file_ops, dataset_name="test_dataset")


@pytest.fixture
def mock_engine_params():
    """Create mock engine parameters for testing."""
    return EngineParams(
        engine_type=EngineType.ANTHROPIC,
        model='claude-3-5-sonnet-20240620',
        config_name='anthropic_claude-3-5-sonnet',
        temperature=0.7,
        max_tokens=1000
    )


@pytest.fixture
def mock_experiment_data():
    """Create mock experiment data for testing."""
    return MockExperimentData()


@pytest.fixture
def sample_openai_tools():
    """Sample OpenAI tool definitions for testing conversion."""
    return [
        {
            "type": "function",
            "function": {
                "name": "add_to_cart",
                "description": "Add a product to the shopping cart",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "product_title": {
                            "type": "string",
                            "description": "The title of the product to add"
                        },
                        "price": {
                            "type": "number",
                            "description": "The price of the product"
                        },
                        "reason": {
                            "type": "string",
                            "description": "Reason for selecting this product"
                        }
                    },
                    "required": ["product_title", "price"]
                }
            }
        },
        {
            "type": "function", 
            "function": {
                "name": "get_product_details",
                "description": "Get detailed information about a product",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "product_id": {
                            "type": "string",
                            "description": "Unique identifier for the product"
                        }
                    },
                    "required": ["product_id"]
                }
            }
        }
    ]


@pytest.fixture
def sample_raw_messages():
    """Sample raw messages in OpenAI format for testing."""
    return [
        {
            "role": "system",
            "content": "You are a helpful shopping assistant."
        },
        {
            "role": "user", 
            "content": "I need help selecting a mousepad from the available options."
        }
    ]


@pytest.fixture
def sample_anthropic_response():
    """Sample Anthropic API response for testing parsing."""
    return {
        "id": "msg_01ABC123DEF456",
        "type": "message",
        "role": "assistant", 
        "model": "claude-3-5-sonnet-20240620",
        "content": [
            {
                "type": "text",
                "text": "I'll help you select a mousepad. Based on the available options, I recommend the SteelSeries QcK Gaming Mouse Pad for its excellent value and performance."
            },
            {
                "type": "tool_use",
                "id": "toolu_01XYZ789ABC123",
                "name": "add_to_cart",
                "input": {
                    "product_title": "SteelSeries QcK Gaming Mouse Pad",
                    "price": 12.99,
                    "reason": "Best value for money with excellent gaming performance"
                }
            }
        ],
        "stop_reason": "tool_use",
        "usage": {
            "input_tokens": 150,
            "output_tokens": 85
        }
    }


@pytest.fixture  
def sample_batch_result_success():
    """Sample successful Anthropic batch result for testing."""
    return {
        "custom_id": "mousepad|baseline|1|anthropic_claude-3-5-sonnet",
        "result": {
            "type": "succeeded",
            "message": {
                "id": "msg_01ABC123DEF456",
                "type": "message",
                "role": "assistant",
                "model": "claude-3-5-sonnet-20240620", 
                "content": [
                    {
                        "type": "text",
                        "text": "I'll help you select a mousepad."
                    },
                    {
                        "type": "tool_use",
                        "id": "toolu_01XYZ789ABC123",
                        "name": "add_to_cart",
                        "input": {
                            "product_title": "SteelSeries QcK Gaming Mouse Pad",
                            "price": 12.99
                        }
                    }
                ],
                "stop_reason": "tool_use",
                "usage": {
                    "input_tokens": 150,
                    "output_tokens": 85
                }
            }
        }
    }


@pytest.fixture
def sample_batch_result_error():
    """Sample error Anthropic batch result for testing."""
    return {
        "custom_id": "mousepad|baseline|1|anthropic_claude-3-5-sonnet",
        "result": {
            "type": "errored",
            "error": {
                "type": "invalid_request_error",
                "message": "Invalid model specified in request"
            }
        }
    }


class TestAnthropicBatchProvider:
    """Test suite for AnthropicBatchProvider class."""
    
    def test_provider_initialization(self, anthropic_provider):
        """Test that the provider initializes correctly."""
        assert isinstance(anthropic_provider, AnthropicBatchProvider)
        assert hasattr(anthropic_provider, 'create_batch_request')
        assert hasattr(anthropic_provider, 'parse_tool_calls_from_response')
        assert hasattr(anthropic_provider, 'extract_response_content')
        assert hasattr(anthropic_provider, 'is_response_successful')
        assert hasattr(anthropic_provider, 'get_error_message')

    def test_convert_openai_tools_to_anthropic(self, anthropic_provider, sample_openai_tools):
        """Test conversion of OpenAI tool schemas to Anthropic format."""
        anthropic_tools = anthropic_provider._convert_openai_tools_to_anthropic(sample_openai_tools)
        
        # Should have same number of tools
        assert len(anthropic_tools) == len(sample_openai_tools)
        
        # Check first tool conversion
        first_tool = anthropic_tools[0]
        assert first_tool['name'] == 'add_to_cart'
        assert first_tool['description'] == 'Add a product to the shopping cart'
        assert 'input_schema' in first_tool
        
        # Check input_schema structure matches Anthropic format
        schema = first_tool['input_schema']
        assert schema['type'] == 'object'
        assert 'properties' in schema
        assert 'product_title' in schema['properties']
        assert 'price' in schema['properties']
        assert schema['required'] == ['product_title', 'price']
        
        # Check second tool
        second_tool = anthropic_tools[1]
        assert second_tool['name'] == 'get_product_details'
        assert second_tool['input_schema']['required'] == ['product_id']

    def test_convert_openai_tools_empty_list(self, anthropic_provider):
        """Test tool conversion with empty tool list."""
        result = anthropic_provider._convert_openai_tools_to_anthropic([])
        assert result == []

    def test_convert_openai_tools_non_function_type(self, anthropic_provider):
        """Test tool conversion filters out non-function types."""
        tools = [
            {"type": "not_function", "function": {"name": "invalid"}},
            {"type": "function", "function": {"name": "valid", "parameters": {"type": "object"}}}
        ]
        result = anthropic_provider._convert_openai_tools_to_anthropic(tools)
        assert len(result) == 1
        assert result[0]['name'] == 'valid'

    def test_create_batch_request_basic(self, anthropic_provider, mock_experiment_data, 
                                       mock_engine_params, sample_raw_messages):
        """Test basic batch request creation according to Anthropic documentation."""
        custom_id = "test_custom_id"
        tools = []
        
        request = anthropic_provider.create_batch_request(
            mock_experiment_data, mock_engine_params, sample_raw_messages, custom_id, tools
        )
        
        # Verify top-level structure matches Anthropic format
        assert 'custom_id' in request
        assert 'params' in request
        # Anthropic provider hashes the custom_id for API compliance
        assert isinstance(request['custom_id'], str)
        assert len(request['custom_id']) == 32  # MD5 hash length
        # Original custom_id should be stored in provider's mapping
        assert request['custom_id'] in anthropic_provider._custom_id_mappings
        assert anthropic_provider._custom_id_mappings[request['custom_id']] == custom_id
        
        # Verify params structure
        params = request['params']
        assert params['model'] == 'claude-3-5-sonnet-20240620'
        assert params['max_tokens'] == 1000
        assert params['temperature'] == 0.7
        assert params['messages'] == sample_raw_messages
        
        # No tools should be present
        assert 'tools' not in params

    def test_create_batch_request_with_tools(self, anthropic_provider, mock_experiment_data,
                                           mock_engine_params, sample_raw_messages, sample_openai_tools):
        """Test batch request creation with tools included."""
        custom_id = "test_with_tools"
        
        request = anthropic_provider.create_batch_request(
            mock_experiment_data, mock_engine_params, sample_raw_messages, custom_id, sample_openai_tools
        )
        
        # Verify tools are converted and included
        params = request['params']
        assert 'tools' in params
        assert len(params['tools']) == 2
        
        # Check tool structure matches Anthropic format
        add_to_cart_tool = params['tools'][0]
        assert add_to_cart_tool['name'] == 'add_to_cart'
        assert 'input_schema' in add_to_cart_tool
        assert add_to_cart_tool['input_schema']['type'] == 'object'

    def test_create_batch_request_minimal_engine_params(self, anthropic_provider, mock_experiment_data,
                                                       sample_raw_messages):
        """Test batch request creation with minimal engine parameters."""
        minimal_params = EngineParams(
            engine_type=EngineType.ANTHROPIC,
            model='claude-3-haiku-20240307',
            config_name='anthropic_claude-3-haiku'
            # No temperature or max_tokens specified
        )
        
        request = anthropic_provider.create_batch_request(
            mock_experiment_data, minimal_params, sample_raw_messages, "minimal_test", []
        )
        
        params = request['params']
        assert params['model'] == 'claude-3-haiku-20240307'
        assert params['max_tokens'] == 1000  # Default value
        # Temperature defaults to 0.0 in EngineParams, so it will be included
        assert params['temperature'] == 0.0

    def test_parse_tool_calls_from_response(self, anthropic_provider, sample_anthropic_response):
        """Test parsing tool calls from Anthropic response format."""
        tool_calls = anthropic_provider.parse_tool_calls_from_response(sample_anthropic_response)
        
        assert len(tool_calls) == 1
        
        tool_call = tool_calls[0]
        assert tool_call['id'] == 'toolu_01XYZ789ABC123'
        assert 'function' in tool_call
        
        function = tool_call['function']
        assert function['name'] == 'add_to_cart'
        
        # Arguments should be JSON string
        arguments = json.loads(function['arguments'])
        assert arguments['product_title'] == 'SteelSeries QcK Gaming Mouse Pad'
        assert arguments['price'] == 12.99
        assert arguments['reason'] == 'Best value for money with excellent gaming performance'

    def test_parse_tool_calls_no_tool_use(self, anthropic_provider):
        """Test parsing when response has no tool_use blocks."""
        response_without_tools = {
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "text", "text": "I can help you, but I need more information."}
            ]
        }
        
        tool_calls = anthropic_provider.parse_tool_calls_from_response(response_without_tools)
        assert tool_calls == []

    def test_parse_tool_calls_empty_content(self, anthropic_provider):
        """Test parsing with empty or missing content."""
        empty_response = {"id": "msg_123", "content": []}
        tool_calls = anthropic_provider.parse_tool_calls_from_response(empty_response)
        assert tool_calls == []
        
        no_content_response = {"id": "msg_123"}
        tool_calls = anthropic_provider.parse_tool_calls_from_response(no_content_response)
        assert tool_calls == []

    def test_parse_tool_calls_multiple_tools(self, anthropic_provider):
        """Test parsing response with multiple tool calls."""
        response_multiple_tools = {
            "content": [
                {"type": "text", "text": "I'll help you with both requests."},
                {
                    "type": "tool_use",
                    "id": "tool1",
                    "name": "add_to_cart", 
                    "input": {"product_title": "Product 1", "price": 10.99}
                },
                {
                    "type": "tool_use",
                    "id": "tool2",
                    "name": "get_product_details",
                    "input": {"product_id": "abc123"}
                }
            ]
        }
        
        tool_calls = anthropic_provider.parse_tool_calls_from_response(response_multiple_tools)
        assert len(tool_calls) == 2
        
        assert tool_calls[0]['function']['name'] == 'add_to_cart'
        assert tool_calls[1]['function']['name'] == 'get_product_details'

    def test_extract_response_content(self, anthropic_provider, sample_anthropic_response):
        """Test extracting text content from Anthropic response."""
        content = anthropic_provider.extract_response_content(sample_anthropic_response)
        
        expected_text = "I'll help you select a mousepad. Based on the available options, I recommend the SteelSeries QcK Gaming Mouse Pad for its excellent value and performance."
        assert content == expected_text

    def test_extract_response_content_multiple_text_blocks(self, anthropic_provider):
        """Test extracting content with multiple text blocks."""
        response_multiple_text = {
            "content": [
                {"type": "text", "text": "First part. "},
                {"type": "tool_use", "id": "tool1", "name": "test", "input": {}},
                {"type": "text", "text": "Second part."}
            ]
        }
        
        content = anthropic_provider.extract_response_content(response_multiple_text)
        assert content == "First part. Second part."

    def test_extract_response_content_no_text(self, anthropic_provider):
        """Test extracting content when there are no text blocks."""
        response_no_text = {
            "content": [
                {"type": "tool_use", "id": "tool1", "name": "test", "input": {}}
            ]
        }
        
        content = anthropic_provider.extract_response_content(response_no_text)
        assert content == ""

    def test_is_response_successful_success(self, anthropic_provider, sample_batch_result_success):
        """Test success detection for successful responses."""
        assert anthropic_provider.is_response_successful(sample_batch_result_success) is True

    def test_is_response_successful_error(self, anthropic_provider, sample_batch_result_error):
        """Test success detection for error responses."""
        assert anthropic_provider.is_response_successful(sample_batch_result_error) is False

    def test_is_response_successful_edge_cases(self, anthropic_provider):
        """Test success detection for edge cases."""
        # Missing result
        no_result = {"custom_id": "test"}
        assert anthropic_provider.is_response_successful(no_result) is False
        
        # Wrong result type
        wrong_type = {
            "result": {"type": "canceled", "message": {}}
        }
        assert anthropic_provider.is_response_successful(wrong_type) is False
        
        # Missing message
        no_message = {
            "result": {"type": "succeeded"}
        }
        assert anthropic_provider.is_response_successful(no_message) is False
        
        # Empty message (should fail because message is empty)
        empty_message = {
            "result": {"type": "succeeded", "message": {}}
        }
        assert anthropic_provider.is_response_successful(empty_message) is False
        
        # Non-empty message (should succeed)
        valid_message = {
            "result": {"type": "succeeded", "message": {"id": "msg_123"}}
        }
        assert anthropic_provider.is_response_successful(valid_message) is True

    def test_get_error_message_errored(self, anthropic_provider, sample_batch_result_error):
        """Test error message extraction for errored responses."""
        error_msg = anthropic_provider.get_error_message(sample_batch_result_error)
        assert "Invalid model specified in request" in error_msg

    def test_get_error_message_different_types(self, anthropic_provider):
        """Test error message extraction for different error types."""
        # Canceled
        canceled_result = {"result": {"type": "canceled"}}
        msg = anthropic_provider.get_error_message(canceled_result)
        assert "canceled" in msg.lower()
        
        # Expired
        expired_result = {"result": {"type": "expired"}}
        msg = anthropic_provider.get_error_message(expired_result)
        assert "expired" in msg.lower()
        
        # Unknown type
        unknown_result = {"result": {"type": "unknown_type"}}
        msg = anthropic_provider.get_error_message(unknown_result)
        assert "unknown_type" in msg

    def test_get_error_message_malformed_error(self, anthropic_provider):
        """Test error message extraction for malformed error responses."""
        # Non-dict error
        malformed = {
            "result": {
                "type": "errored",
                "error": "String error instead of dict"
            }
        }
        msg = anthropic_provider.get_error_message(malformed)
        assert "String error instead of dict" in msg

    def test_batch_request_format_compliance(self, anthropic_provider, mock_experiment_data,
                                           mock_engine_params, sample_raw_messages, sample_openai_tools):
        """Test that batch requests comply exactly with Anthropic documentation format."""
        custom_id = "compliance_test"
        
        request = anthropic_provider.create_batch_request(
            mock_experiment_data, mock_engine_params, sample_raw_messages, custom_id, sample_openai_tools
        )
        
        # Verify exact structure from Anthropic docs (includes _original_custom_id for internal tracking)
        assert set(request.keys()) == {"custom_id", "params", "_original_custom_id"}
        
        params = request["params"]
        required_params = {"model", "max_tokens", "messages"}
        assert required_params.issubset(set(params.keys()))
        
        # Verify messages format (should be unchanged from input)
        assert params["messages"] == sample_raw_messages
        
        # Verify tools format if present
        if "tools" in params:
            for tool in params["tools"]:
                assert set(tool.keys()) == {"name", "description", "input_schema"}
                assert tool["input_schema"]["type"] == "object"

    def test_tool_call_parsing_compliance(self, anthropic_provider):
        """Test that tool call parsing handles all documented Anthropic response formats."""
        # Test with minimal tool_use block
        minimal_response = {
            "content": [
                {
                    "type": "tool_use",
                    "id": "minimal_id",
                    "name": "minimal_tool",
                    "input": {}
                }
            ]
        }
        
        tool_calls = anthropic_provider.parse_tool_calls_from_response(minimal_response)
        assert len(tool_calls) == 1
        assert tool_calls[0]["id"] == "minimal_id"
        assert tool_calls[0]["function"]["name"] == "minimal_tool"
        assert json.loads(tool_calls[0]["function"]["arguments"]) == {}

    def test_error_handling_robustness(self, anthropic_provider):
        """Test that error handling is robust against malformed inputs."""
        # Test with completely empty input
        assert anthropic_provider.parse_tool_calls_from_response({}) == []
        assert anthropic_provider.extract_response_content({}) == ""
        assert anthropic_provider.is_response_successful({}) is False
        
        # Test with None values
        assert anthropic_provider.parse_tool_calls_from_response({"content": None}) == []
        
        # Test get_error_message with missing structure
        error_msg = anthropic_provider.get_error_message({})
        assert "Request failed with type: unknown" in error_msg

    def test_integration_workflow(self, anthropic_provider, mock_experiment_data, 
                                 mock_engine_params, sample_raw_messages, sample_openai_tools):
        """Test a complete workflow from request creation to response parsing."""
        custom_id = "integration_test"
        
        # Step 1: Create batch request
        request = anthropic_provider.create_batch_request(
            mock_experiment_data, mock_engine_params, sample_raw_messages, custom_id, sample_openai_tools
        )
        
        # Verify request structure (custom_id is hashed for API compliance)
        assert isinstance(request["custom_id"], str)
        assert len(request["custom_id"]) == 32  # MD5 hash
        assert request["_original_custom_id"] == custom_id
        assert "tools" in request["params"]
        
        # Step 2: Simulate successful response
        mock_response = {
            "custom_id": custom_id,
            "result": {
                "type": "succeeded",
                "message": {
                    "content": [
                        {"type": "text", "text": "I recommend this product:"},
                        {
                            "type": "tool_use",
                            "id": "final_selection",
                            "name": "add_to_cart",
                            "input": {
                                "product_title": "SteelSeries QcK Gaming Mouse Pad",
                                "price": 12.99
                            }
                        }
                    ]
                }
            }
        }
        
        # Step 3: Verify response processing
        assert anthropic_provider.is_response_successful(mock_response) is True
        
        tool_calls = anthropic_provider.parse_tool_calls_from_response(mock_response["result"]["message"])
        assert len(tool_calls) == 1
        assert tool_calls[0]["function"]["name"] == "add_to_cart"
        
        content = anthropic_provider.extract_response_content(mock_response["result"]["message"])
        assert "I recommend this product:" in content

    def test_custom_id_handling(self, anthropic_provider, mock_experiment_data,
                               mock_engine_params, sample_raw_messages):
        """Test that custom_id is properly handled and preserved."""
        test_custom_ids = [
            "simple_id",
            "complex|id|with|pipes",
            "mousepad|baseline|1|anthropic_claude-3-5-sonnet",
            "special-chars_123!@#"
        ]
        
        for custom_id in test_custom_ids:
            request = anthropic_provider.create_batch_request(
                mock_experiment_data, mock_engine_params, sample_raw_messages, custom_id, []
            )
            # Custom ID is hashed for API compliance, but original is preserved
            assert isinstance(request["custom_id"], str)
            assert len(request["custom_id"]) == 32  # MD5 hash
            assert request["_original_custom_id"] == custom_id


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v"])