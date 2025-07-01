#!/usr/bin/env python3
"""
Test Anthropic Batch API with actual image processing.
This test demonstrates using the dummy image with the Anthropic batch provider
to ensure image-based requests work correctly.
"""

import base64
import json
import pytest
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd

# Import the modules we're testing
from experiments.runners.batch_runtime.providers.anthropic import AnthropicBatchProvider
from agent.src.typedefs import EngineParams, EngineType


class MockExperimentDataWithImage:
    """Mock ExperimentData with image for testing."""
    
    def __init__(self, query: str = "mousepad", experiment_label: str = "baseline", 
                 experiment_number: int = 1, image_path: Path = None):
        self.query = query
        self.experiment_label = experiment_label
        self.experiment_number = experiment_number
        self.experiment_df = pd.DataFrame({
            'product_title': ['SteelSeries QcK Gaming Mouse Pad', 'Corsair MM300'],
            'price': [12.99, 24.99],
            'rating': [4.2, 4.8]
        })
        self.image_path = image_path


@pytest.fixture
def test_image_path():
    """Path to the test image."""
    return Path(__file__).parent / "fixtures" / "test_screenshot.png"


@pytest.fixture
def anthropic_provider():
    """Create an AnthropicBatchProvider instance for testing."""
    return AnthropicBatchProvider()


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
def sample_tools_with_screenshot():
    """Sample tools that work with screenshot analysis."""
    return [
        {
            "type": "function",
            "function": {
                "name": "add_to_cart",
                "description": "Add a product to the shopping cart based on screenshot analysis",
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
                        "screenshot_analysis": {
                            "type": "string",
                            "description": "Analysis of what was visible in the screenshot"
                        }
                    },
                    "required": ["product_title", "price"]
                }
            }
        }
    ]


def encode_image_to_base64(image_path: Path) -> str:
    """Encode an image file to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


@pytest.fixture
def sample_messages_with_image(test_image_path):
    """Sample messages including an image in OpenAI format."""
    if not test_image_path.exists():
        pytest.skip(f"Test image not found at {test_image_path}")
    
    base64_image = encode_image_to_base64(test_image_path)
    
    return [
        {
            "role": "system",
            "content": "You are a helpful shopping assistant that can analyze product screenshots."
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Please analyze this shopping website screenshot and help me select the best mousepad based on what you can see."
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}"
                    }
                }
            ]
        }
    ]


class TestAnthropicBatchWithImage:
    """Test suite for Anthropic Batch API with image processing."""
    
    def test_image_file_exists(self, test_image_path):
        """Verify that the test image file exists and is readable."""
        assert test_image_path.exists(), f"Test image not found at {test_image_path}"
        assert test_image_path.is_file(), f"Test image path is not a file: {test_image_path}"
        assert test_image_path.suffix.lower() == '.png', f"Expected PNG file, got {test_image_path.suffix}"
        
        # Verify file is not empty
        assert test_image_path.stat().st_size > 0, "Test image file is empty"
        
        # Verify we can read the image
        with open(test_image_path, 'rb') as f:
            image_data = f.read()
            assert len(image_data) > 1000, "Image file seems too small to be a valid screenshot"
    
    def test_create_batch_request_with_image_content(self, anthropic_provider, mock_engine_params,
                                                   sample_messages_with_image, sample_tools_with_screenshot,
                                                   test_image_path):
        """Test creating batch request with image content in messages."""
        mock_data = MockExperimentDataWithImage(image_path=test_image_path)
        custom_id = "image_test_001"
        
        request = anthropic_provider.create_batch_request(
            mock_data, mock_engine_params, sample_messages_with_image, custom_id, sample_tools_with_screenshot
        )
        
        # Verify basic request structure (custom_id is hashed for API compliance)
        assert isinstance(request['custom_id'], str)
        assert len(request['custom_id']) == 32  # MD5 hash
        assert request.get('_original_custom_id') == custom_id
        assert 'params' in request
        
        params = request['params']
        assert params['model'] == 'claude-3-5-sonnet-20240620'
        assert 'messages' in params
        assert 'tools' in params
        
        # Verify messages structure is preserved
        messages = params['messages']
        assert len(messages) == 2
        assert messages[0]['role'] == 'system'
        assert messages[1]['role'] == 'user'
        
        # Verify user message has multipart content
        user_content = messages[1]['content']
        assert isinstance(user_content, list)
        assert len(user_content) == 2
        
        # Check text part
        text_part = user_content[0]
        assert text_part['type'] == 'text'
        assert 'analyze this shopping website screenshot' in text_part['text'].lower()
        
        # Check image part
        image_part = user_content[1]
        assert image_part['type'] == 'image_url'
        assert 'image_url' in image_part
        assert 'url' in image_part['image_url']
        assert image_part['image_url']['url'].startswith('data:image/png;base64,')
    
    def test_tool_conversion_with_screenshot_tools(self, anthropic_provider, sample_tools_with_screenshot):
        """Test that tools with screenshot analysis capabilities are properly converted."""
        anthropic_tools = anthropic_provider._convert_openai_tools_to_anthropic(sample_tools_with_screenshot)
        
        assert len(anthropic_tools) == 1
        
        tool = anthropic_tools[0]
        assert tool['name'] == 'add_to_cart'
        assert 'screenshot' in tool['description'].lower()
        
        schema = tool['input_schema']
        assert 'screenshot_analysis' in schema['properties']
        assert schema['properties']['screenshot_analysis']['type'] == 'string'
    
    def test_parse_screenshot_analysis_response(self, anthropic_provider):
        """Test parsing response that includes screenshot analysis."""
        mock_response_with_analysis = {
            "id": "msg_screenshot_analysis",
            "type": "message",
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "I can see this is a shopping website showing mousepad search results. Based on the screenshot analysis, I recommend the SteelSeries QcK Gaming Mouse Pad which appears to offer the best value."
                },
                {
                    "type": "tool_use",
                    "id": "toolu_screenshot_001",
                    "name": "add_to_cart",
                    "input": {
                        "product_title": "SteelSeries QcK Gaming Mouse Pad",
                        "price": 12.99,
                        "screenshot_analysis": "Screenshot shows 4 mousepad products in a grid layout. The SteelSeries QcK is prominently displayed with good ratings and competitive pricing at $12.99."
                    }
                }
            ]
        }
        
        # Test content extraction
        content = anthropic_provider.extract_response_content(mock_response_with_analysis)
        assert "screenshot analysis" in content.lower()
        assert "steelseries qck" in content.lower()
        
        # Test tool call parsing
        tool_calls = anthropic_provider.parse_tool_calls_from_response(mock_response_with_analysis)
        assert len(tool_calls) == 1
        
        tool_call = tool_calls[0]
        assert tool_call['function']['name'] == 'add_to_cart'
        
        arguments = json.loads(tool_call['function']['arguments'])
        assert 'screenshot_analysis' in arguments
        assert "grid layout" in arguments['screenshot_analysis']
        assert arguments['product_title'] == "SteelSeries QcK Gaming Mouse Pad"
        assert arguments['price'] == 12.99
    
    def test_batch_format_compliance_with_image(self, anthropic_provider, mock_engine_params,
                                              sample_messages_with_image, sample_tools_with_screenshot,
                                              test_image_path):
        """Test that batch requests with images still comply with Anthropic documentation format."""
        mock_data = MockExperimentDataWithImage(image_path=test_image_path)
        
        request = anthropic_provider.create_batch_request(
            mock_data, mock_engine_params, sample_messages_with_image, "compliance_image_test", sample_tools_with_screenshot
        )
        
        # Verify exact structure requirements (includes _original_custom_id for internal tracking)
        assert set(request.keys()) == {"custom_id", "params", "_original_custom_id"}
        
        params = request["params"]
        required_params = {"model", "max_tokens", "messages"}
        assert required_params.issubset(set(params.keys()))
        
        # Verify that image content doesn't break the format
        assert isinstance(params["messages"], list)
        assert all(isinstance(msg, dict) for msg in params["messages"])
        
        # Verify tools are properly formatted
        if "tools" in params:
            for tool in params["tools"]:
                assert set(tool.keys()) == {"name", "description", "input_schema"}
    
    def test_image_data_integrity(self, test_image_path):
        """Test that image encoding/decoding preserves data integrity."""
        # Read original image
        with open(test_image_path, 'rb') as f:
            original_data = f.read()
        
        # Encode to base64
        base64_data = encode_image_to_base64(test_image_path)
        
        # Decode back to bytes
        decoded_data = base64.b64decode(base64_data)
        
        # Verify integrity
        assert len(decoded_data) == len(original_data)
        assert decoded_data == original_data
        assert len(base64_data) > 0
        
        # Verify base64 format
        assert all(c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/='
                  for c in base64_data)
    
    def test_integration_with_real_screenshot_workflow(self, anthropic_provider, mock_engine_params,
                                                     sample_messages_with_image, sample_tools_with_screenshot,
                                                     test_image_path):
        """Test complete workflow simulating real screenshot-based shopping assistance."""
        mock_data = MockExperimentDataWithImage(
            query="mousepad",
            experiment_label="screenshot_analysis",
            experiment_number=1,
            image_path=test_image_path
        )
        
        # Step 1: Create batch request with image
        custom_id = f"{mock_data.query}|{mock_data.experiment_label}|{mock_data.experiment_number}|anthropic_claude-3-5-sonnet"
        
        request = anthropic_provider.create_batch_request(
            mock_data, mock_engine_params, sample_messages_with_image, custom_id, sample_tools_with_screenshot
        )
        
        # Verify request structure (custom_id is hashed for API compliance)
        assert isinstance(request["custom_id"], str)
        assert len(request["custom_id"]) == 32  # MD5 hash
        assert request["_original_custom_id"] == custom_id
        assert "tools" in request["params"]
        
        # Step 2: Simulate successful response with screenshot analysis
        mock_successful_response = {
            "custom_id": custom_id,
            "result": {
                "type": "succeeded", 
                "message": {
                    "id": "msg_real_workflow",
                    "content": [
                        {
                            "type": "text",
                            "text": "Based on the screenshot showing mousepad search results, I can see several options with different price points and features."
                        },
                        {
                            "type": "tool_use",
                            "id": "toolu_workflow_001",
                            "name": "add_to_cart",
                            "input": {
                                "product_title": "SteelSeries QcK Gaming Mouse Pad",
                                "price": 12.99,
                                "screenshot_analysis": "The screenshot displays a clean shopping interface with 4 mousepad options. The SteelSeries QcK appears at the top with strong ratings and competitive pricing."
                            }
                        }
                    ]
                }
            }
        }
        
        # Step 3: Verify response processing
        assert anthropic_provider.is_response_successful(mock_successful_response) is True
        
        tool_calls = anthropic_provider.parse_tool_calls_from_response(mock_successful_response["result"]["message"])
        assert len(tool_calls) == 1
        
        # Verify the tool call contains screenshot analysis
        arguments = json.loads(tool_calls[0]["function"]["arguments"])
        assert "screenshot_analysis" in arguments
        assert "screenshot" in arguments["screenshot_analysis"].lower()
        
        # Step 4: Verify content extraction includes screenshot analysis
        content = anthropic_provider.extract_response_content(mock_successful_response["result"]["message"])
        assert "screenshot" in content.lower()
        assert "mousepad" in content.lower()


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v"])