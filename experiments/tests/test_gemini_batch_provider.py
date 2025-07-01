#!/usr/bin/env python3
"""
Comprehensive pytest suite for Gemini/Vertex AI Batch API implementation.
Tests ensure compliance with Google Cloud AI Platform documentation and proper functioning
of all batch provider methods.
"""

import json
import pytest
import tempfile
from pathlib import Path
from typing import Dict, List, Any
from unittest.mock import Mock, patch, MagicMock
import pandas as pd

# Mock the Google Cloud dependencies before importing
import sys
mock_vertexai = Mock()
mock_aiplatform = Mock()
mock_storage = Mock()

sys.modules['vertexai'] = mock_vertexai
sys.modules['google.cloud'] = Mock()
sys.modules['google.cloud.storage'] = mock_storage  
sys.modules['google.cloud.aiplatform'] = mock_aiplatform

# Import the modules we're testing
from experiments.runners.batch_runtime.providers.gemini import GeminiBatchProvider
from experiments.runners.batch_runtime.services.file_operations import FileOperationsService
from agent.src.typedefs import EngineParams, EngineType


class MockExperimentData:
    """Mock ExperimentData for testing."""
    
    def __init__(self, query: str = "mousepad", experiment_label: str = "baseline", 
                 experiment_number: int = 1):
        self.query = query
        self.experiment_label = experiment_label
        self.experiment_number = experiment_number
        self.experiment_id = f"{query}_{experiment_label}_{experiment_number}"
        self.experiment_df = pd.DataFrame({
            'product_title': ['SteelSeries QcK Gaming Mouse Pad', 'Corsair MM300'],
            'price': [12.99, 24.99],
            'rating': [4.2, 4.8]
        })


@pytest.fixture
def mock_gcs_environment():
    """Set up mock environment variables for GCS."""
    import os
    original_env = {}
    
    test_env = {
        'GCS_BUCKET_NAME': 'test-bucket',
        'GOOGLE_CLOUD_PROJECT': 'test-project',
        'GOOGLE_CLOUD_LOCATION': 'us-central1'
    }
    
    # Store original values
    for key in test_env:
        if key in os.environ:
            original_env[key] = os.environ[key]
        os.environ[key] = test_env[key]
    
    yield test_env
    
    # Restore original values
    for key in test_env:
        if key in original_env:
            os.environ[key] = original_env[key]
        elif key in os.environ:
            del os.environ[key]


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
def gemini_provider(mock_gcs_environment, mock_file_ops):
    """Create a GeminiBatchProvider instance for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Mock the storage client and other dependencies
        mock_storage_client = Mock()
        mock_bucket = Mock()
        mock_blob = Mock()
        mock_bucket.blob.return_value = mock_blob
        mock_storage_client.bucket.return_value = mock_bucket
        mock_storage.Client.return_value = mock_storage_client
        
        provider = GeminiBatchProvider(
            file_ops=mock_file_ops,
            screenshots_dir=temp_path / 'screenshots',
            project_id='test-project',
            location='us-central1',
            bucket_name='test-bucket',
            dataset_name='test-dataset'
        )
        
        # Override the storage client with our mock
        if hasattr(provider, 'storage_client'):
            provider.storage_client = mock_storage_client
        
        yield provider


@pytest.fixture
def mock_engine_params():
    """Create mock engine parameters for testing."""
    return EngineParams(
        engine_type=EngineType.GEMINI,
        model='gemini-2.5-flash',
        config_name='gemini_gemini-2.5-flash',
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
        }
    ]


@pytest.fixture
def sample_raw_messages_with_images():
    """Sample raw messages with image URLs for testing."""
    return [
        {
            "role": "system",
            "content": "You are a helpful shopping assistant."
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Help me select a mousepad from these options:"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://storage.googleapis.com/test-bucket/screenshots/mousepad_baseline_1.png"
                    }
                }
            ]
        }
    ]


@pytest.fixture
def sample_raw_messages_text_only():
    """Sample raw messages without images for testing."""
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
def sample_vertex_batch_response():
    """Sample Vertex AI batch prediction response for testing."""
    return {
        "prediction": {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": "I'll help you select a mousepad. Based on the options, I recommend the SteelSeries QcK Gaming Mouse Pad."
                            },
                            {
                                "functionCall": {
                                    "name": "add_to_cart",
                                    "args": {
                                        "product_title": "SteelSeries QcK Gaming Mouse Pad",
                                        "price": 12.99,
                                        "reason": "Best value for money"
                                    }
                                }
                            }
                        ]
                    },
                    "finishReason": "STOP"
                }
            ]
        }
    }


@pytest.fixture
def sample_vertex_error_response():
    """Sample Vertex AI error response for testing."""
    return {
        "error": {
            "code": 400,
            "message": "Invalid model specified",
            "status": "INVALID_ARGUMENT"
        }
    }


class TestGeminiBatchProvider:
    """Test suite for GeminiBatchProvider class."""
    
    def test_provider_initialization(self, gemini_provider):
        """Test that the provider initializes correctly."""
        assert isinstance(gemini_provider, GeminiBatchProvider)
        assert hasattr(gemini_provider, 'create_batch_request')
        assert hasattr(gemini_provider, 'parse_tool_calls_from_response')
        assert hasattr(gemini_provider, 'extract_response_content')
        assert hasattr(gemini_provider, 'is_response_successful')
        assert hasattr(gemini_provider, 'get_error_message')
        assert hasattr(gemini_provider, 'upload_and_submit_batches')
        
        # Verify configuration is accessible through config object
        assert hasattr(gemini_provider, 'config')
        assert gemini_provider.config.bucket_name == 'test-bucket'
        assert gemini_provider.config.project_id == 'test-project'
        assert gemini_provider.config.location == 'us-central1'

    def test_gcs_uri_generation(self, gemini_provider):
        """Test GCS URI generation from local paths."""
        # Test normal path structure (datasets/screenshots structure)
        local_path = "/Users/test/datasets/screenshots/mousepad/baseline/mousepad_baseline_1.png"
        gcs_uri = gemini_provider._get_gcs_uri(local_path)
        
        # The implementation maps to: screenshots/{dataset_name}/{query}/{experiment_label}/{filename}
        expected_uri = "gs://test-bucket/screenshots/mousepad/baseline/mousepad_baseline_1.png"
        assert gcs_uri == expected_uri
        
        # Test fallback for non-standard paths
        local_path = "/some/other/path/screenshot.png"
        gcs_uri = gemini_provider._get_gcs_uri(local_path)
        
        expected_uri = "gs://test-bucket/screenshots/screenshot.png"
        assert gcs_uri == expected_uri

    def test_create_batch_request_text_only(self, gemini_provider, mock_experiment_data,
                                           mock_engine_params, sample_raw_messages_text_only):
        """Test batch request creation with text-only messages."""
        custom_id = "test_text_only"
        tools = []
        
        request = gemini_provider.create_batch_request(
            mock_experiment_data, mock_engine_params, sample_raw_messages_text_only, custom_id, tools
        )
        
        # Verify Gemini GenerateContentRequest structure
        assert 'contents' in request
        assert 'model' in request
        assert 'custom_id' in request
        assert request['model'] == 'gemini-2.5-flash'
        assert request['custom_id'] == custom_id
        
        # Verify contents structure - system message should be extracted to systemInstruction
        contents = request['contents']
        assert len(contents) == 1  # Only user message (system message converted to systemInstruction)
        
        # Check system instruction
        assert 'system_instruction' in request
        assert request['system_instruction'][0]['parts'][0]['text'] == 'You are a helpful shopping assistant.'
        
        # Check user message
        assert contents[0]['role'] == 'user'
        assert contents[0]['parts'][0]['text'] == 'I need help selecting a mousepad from the available options.'

    def test_create_batch_request_with_images(self, gemini_provider, mock_experiment_data,
                                            mock_engine_params, sample_raw_messages_with_images):
        """Test batch request creation with image URLs."""
        custom_id = "test_with_images"
        tools = []
        
        request = gemini_provider.create_batch_request(
            mock_experiment_data, mock_engine_params, sample_raw_messages_with_images, custom_id, tools
        )
        
        # Verify custom_id is included
        assert 'custom_id' in request
        assert request['custom_id'] == custom_id
        
        # Verify contents structure - system message should be extracted to systemInstruction
        contents = request['contents']
        assert len(contents) == 1  # Only user message (system message converted to systemInstruction)
        
        # Check system instruction
        assert 'system_instruction' in request
        assert request['system_instruction'][0]['parts'][0]['text'] == 'You are a helpful shopping assistant.'
        
        # Check user message with image
        user_message = contents[0]  # Now the first (and only) message in contents
        assert user_message['role'] == 'user'
        assert len(user_message['parts']) == 2  # Text + image
        
        # Check text part
        assert user_message['parts'][0]['text'] == 'Help me select a mousepad from these options:'
        
        # Check image part - should be converted to GCS URI format
        image_part = user_message['parts'][1]
        assert 'file_data' in image_part
        assert image_part['file_data']['mime_type'] == 'image/png'
        assert image_part['file_data']['file_uri'] == 'gs://test-bucket/screenshots/mousepad_baseline_1.png'

    def test_create_batch_request_with_tools(self, gemini_provider, mock_experiment_data,
                                           mock_engine_params, sample_raw_messages_text_only, sample_openai_tools):
        """Test batch request creation with tools included."""
        custom_id = "test_with_tools"
        
        request = gemini_provider.create_batch_request(
            mock_experiment_data, mock_engine_params, sample_raw_messages_text_only, custom_id, sample_openai_tools
        )
        
        # Verify tools are converted and included
        assert 'tools' in request
        assert len(request['tools']) == 1
        
        # Check tool structure matches Gemini format
        tool = request['tools'][0]
        assert 'function_declarations' in tool
        assert len(tool['function_declarations']) == 1
        
        function_decl = tool['function_declarations'][0]
        assert function_decl['name'] == 'add_to_cart'
        assert function_decl['description'] == 'Add a product to the shopping cart'
        assert 'parameters' in function_decl

    def test_create_batch_request_base64_image_rejection(self, gemini_provider, mock_experiment_data,
                                                       mock_engine_params):
        """Test that base64 images are properly rejected."""
        messages_with_base64 = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Test message"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
                        }
                    }
                ]
            }
        ]
        
        request = gemini_provider.create_batch_request(
            mock_experiment_data, mock_engine_params, messages_with_base64, "test_base64", []
        )
        
        # Base64 image should be skipped, only text should remain
        contents = request['contents']
        assert len(contents) == 1
        user_message = contents[0]
        assert len(user_message['parts']) == 1  # Only text part
        assert user_message['parts'][0]['text'] == 'Test message'

    def test_batch_prediction_job_parameters(self, gemini_provider):
        """Test that genai client batch creation is used correctly in source code."""
        # This test verifies that we're now using the genai client instead of aiplatform
        
        import inspect
        source = inspect.getsource(gemini_provider._submit_single_batch)
        
        # Verify that we're using genai client instead of aiplatform
        assert 'self.genai_client.batches.create(' in source, "Should use genai client for batch creation"
        assert 'BatchPredictionJob.create(' not in source, "Should not use aiplatform.BatchPredictionJob.create anymore"
        
        # Verify that the genai client parameters are present
        genai_params = [
            'model=',
            'src=',
            'config=CreateBatchJobConfig'
        ]
        
        for param in genai_params:
            assert param in source, f"Missing required genai parameter '{param}' in source code"
        
        # Verify the destination is configured correctly
        assert 'dest=gcs_output_uri' in source, "Should set destination URI in CreateBatchJobConfig"
        
        # Verify that old problematic parameters are NOT present
        problematic_params = ['input_config', 'output_config', 'instances_format', 'predictions_format']
        for param in problematic_params:
            assert param not in source, f"Found old parameter '{param}' - should use genai client instead"

    def test_jsonl_format_includes_request_property(self, gemini_provider):
        """Test that JSONL format includes required 'request' property and custom_id."""
        import inspect
        source = inspect.getsource(gemini_provider._submit_single_batch)
        
        # Verify that we wrap requests in the required "request" property
        assert '"request": req' in source, "JSONL format must include 'request' property wrapper"
        assert 'batch_item = {' in source, "Must create batch_item dict"
        assert '"custom_id": custom_id' in source, "JSONL format must include 'custom_id' property"
        assert 'req.pop(\'custom_id\', None)' in source, "Must extract custom_id from request body"
        
    def test_vertex_api_output_suppression(self, gemini_provider):
        """Test that genai client API output is suppressed to prevent progress bar interference."""
        import inspect
        source = inspect.getsource(gemini_provider._submit_single_batch)
        
        # Verify that we suppress API output
        assert 'redirect_stdout' in source, "Must suppress stdout to prevent progress bar interference"
        assert 'redirect_stderr' in source, "Must suppress stderr to prevent progress bar interference"
        assert 'logging.getLogger' in source, "Must suppress logging output"

    def test_parse_tool_calls_from_response(self, gemini_provider, sample_vertex_batch_response):
        """Test parsing tool calls from Vertex AI response format."""
        tool_calls = gemini_provider.parse_tool_calls_from_response(sample_vertex_batch_response)
        
        assert len(tool_calls) == 1
        
        tool_call = tool_calls[0]
        assert 'function' in tool_call
        
        function = tool_call['function']
        assert function['name'] == 'add_to_cart'
        
        # Arguments should be JSON string
        arguments = json.loads(function['arguments'])
        assert arguments['product_title'] == 'SteelSeries QcK Gaming Mouse Pad'
        assert arguments['price'] == 12.99
        assert arguments['reason'] == 'Best value for money'

    def test_parse_tool_calls_no_function_calls(self, gemini_provider):
        """Test parsing when response has no function calls."""
        response_without_tools = {
            "prediction": {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {"text": "I can help you, but I need more information."}
                            ]
                        }
                    }
                ]
            }
        }
        
        tool_calls = gemini_provider.parse_tool_calls_from_response(response_without_tools)
        assert tool_calls == []

    def test_extract_response_content(self, gemini_provider, sample_vertex_batch_response):
        """Test extracting text content from Vertex AI response."""
        content = gemini_provider.extract_response_content(sample_vertex_batch_response)
        
        expected_text = "I'll help you select a mousepad. Based on the options, I recommend the SteelSeries QcK Gaming Mouse Pad."
        assert content == expected_text

    def test_extract_response_content_no_text(self, gemini_provider):
        """Test extracting content when there are no text parts."""
        response_no_text = {
            "prediction": {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "functionCall": {
                                        "name": "test_function",
                                        "args": {}
                                    }
                                }
                            ]
                        }
                    }
                ]
            }
        }
        
        content = gemini_provider.extract_response_content(response_no_text)
        assert content == ""

    def test_is_response_successful_success(self, gemini_provider, sample_vertex_batch_response):
        """Test success detection for successful responses."""
        assert gemini_provider.is_response_successful(sample_vertex_batch_response) is True

    def test_is_response_successful_error(self, gemini_provider, sample_vertex_error_response):
        """Test success detection for error responses."""
        assert gemini_provider.is_response_successful(sample_vertex_error_response) is False

    def test_is_response_successful_edge_cases(self, gemini_provider):
        """Test success detection for edge cases."""
        # Missing prediction
        no_prediction = {}
        assert gemini_provider.is_response_successful(no_prediction) is False
        
        # Empty candidates
        empty_candidates = {
            "prediction": {"candidates": []}
        }
        assert gemini_provider.is_response_successful(empty_candidates) is False
        
        # No content in candidate
        no_content = {
            "prediction": {
                "candidates": [{"finishReason": "STOP"}]
            }
        }
        assert gemini_provider.is_response_successful(no_content) is False

    def test_get_error_message_error_response(self, gemini_provider, sample_vertex_error_response):
        """Test error message extraction for error responses."""
        error_msg = gemini_provider.get_error_message(sample_vertex_error_response)
        assert "Invalid model specified" in error_msg

    def test_get_error_message_candidate_error(self, gemini_provider):
        """Test error message extraction for candidate-level errors."""
        candidate_error = {
            "prediction": {
                "candidates": [
                    {
                        "finishReason": "SAFETY",
                        "content": {"parts": []}
                    }
                ]
            }
        }
        
        error_msg = gemini_provider.get_error_message(candidate_error)
        assert "SAFETY" in error_msg

    def test_get_response_body_from_result(self, gemini_provider, sample_vertex_batch_response):
        """Test extracting response body from batch result."""
        response_body = gemini_provider.get_response_body_from_result(sample_vertex_batch_response)
        
        # Should return the prediction part
        assert response_body == sample_vertex_batch_response['prediction']

    def test_integration_workflow(self, gemini_provider, mock_experiment_data, 
                                 mock_engine_params, sample_raw_messages_with_images, sample_openai_tools):
        """Test a complete workflow from request creation to response parsing."""
        custom_id = "integration_test"
        
        # Step 1: Create batch request
        request = gemini_provider.create_batch_request(
            mock_experiment_data, mock_engine_params, sample_raw_messages_with_images, custom_id, sample_openai_tools
        )
        
        # Verify request structure
        assert 'contents' in request
        assert 'model' in request
        assert 'tools' in request
        assert request['model'] == 'gemini-2.5-flash'
        
        # Step 2: Simulate successful response
        mock_response = {
            "prediction": {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {"text": "I recommend this product:"},
                                {
                                    "functionCall": {
                                        "name": "add_to_cart",
                                        "args": {
                                            "product_title": "SteelSeries QcK Gaming Mouse Pad",
                                            "price": 12.99
                                        }
                                    }
                                }
                            ]
                        }
                    }
                ]
            }
        }
        
        # Step 3: Verify response processing
        assert gemini_provider.is_response_successful(mock_response) is True
        
        tool_calls = gemini_provider.parse_tool_calls_from_response(mock_response)
        assert len(tool_calls) == 1
        assert tool_calls[0]['function']['name'] == 'add_to_cart'
        
        content = gemini_provider.extract_response_content(mock_response)
        assert "I recommend this product:" in content

    def test_error_handling_robustness(self, gemini_provider):
        """Test that error handling is robust against malformed inputs."""
        # Test with completely empty input
        assert gemini_provider.parse_tool_calls_from_response({}) == []
        assert gemini_provider.extract_response_content({}) == ""
        assert gemini_provider.is_response_successful({}) is False
        
        # Test with None values
        assert gemini_provider.parse_tool_calls_from_response({"prediction": None}) == []
        
        # Test get_error_message with missing structure
        error_msg = gemini_provider.get_error_message({})
        assert "Unknown error" in error_msg

    def test_vertex_ai_api_compliance(self, gemini_provider):
        """Test that the implementation uses genai client correctly."""
        
        # Test that we're using the genai client instead of the old aiplatform API
        
        # Read the source code to verify the implementation
        import inspect
        source = inspect.getsource(gemini_provider._submit_single_batch)
        
        # Verify we're using genai client
        assert 'self.genai_client.batches.create(' in source, "Should use genai client for batch creation"
        assert 'BatchPredictionJob.create(' not in source, "Should not use aiplatform API anymore"
        
        # Verify genai-specific parameters are present
        genai_params = ['model=', 'src=', 'CreateBatchJobConfig']
        for param in genai_params:
            assert param in source, f"Required genai parameter '{param}' not found in _submit_single_batch method"
        
        # Verify old problematic parameters are NOT present
        old_params = ['instances_format', 'predictions_format', 'input_config', 'output_config']
        for param in old_params:
            assert param not in source, f"Old parameter '{param}' found - should use genai client instead"

    def test_batch_size_grouping(self, gemini_provider):
        """Test that experiments are grouped into batches of 100."""
        import inspect
        source = inspect.getsource(gemini_provider.upload_and_submit_batches)
        
        # Verify that we use 100-experiment batches instead of label-based grouping
        assert 'batch_size = 100' in source, "Must group experiments into batches of 100"
        assert 'range(0, total_experiments, batch_size)' in source, "Must iterate through experiments in chunks of batch_size"
        assert 'batch_config_name = f"{config_name}_batch_{batch_number}"' in source, "Must name batches with batch numbers"
        
        # Verify that we don't use label-based grouping anymore
        assert 'experiments_by_label' not in source, "Should not group by experiment label anymore"
        assert 'defaultdict' not in source, "Should not use defaultdict for label grouping anymore"

    def test_list_batches_functionality(self, gemini_provider):
        """Test that list_batches method uses genai client correctly."""
        import inspect
        source = inspect.getsource(gemini_provider.list_batches)
        
        # Verify that we're using genai client for listing
        assert 'self.genai_client.batches.list(' in source, "Should use genai client for batch listing"
        assert 'ListBatchJobsConfig' in source, "Should use ListBatchJobsConfig for configuration"
        assert 'page_size=' in source, "Should support pagination"
        
        # Verify method signature and parameters
        assert 'page_size: int = 10' in source, "Should have page_size parameter with default"
        assert 'max_batches: Optional[int] = None' in source, "Should have max_batches parameter"

    def test_get_batch_by_name_functionality(self, gemini_provider):
        """Test that get_batch_by_name method uses genai client correctly."""
        import inspect
        source = inspect.getsource(gemini_provider.get_batch_by_name)
        
        # Verify that we're using genai client for getting specific batches
        assert 'self.genai_client.batches.get(' in source, "Should use genai client for getting batches"
        assert 'name=batch_name' in source, "Should pass batch_name as name parameter"
        
        # Verify method signature
        assert 'batch_name: str' in source, "Should have batch_name parameter"
        assert 'Optional[Dict[str, Any]]' in source, "Should return optional dict"

    def test_delete_batch_functionality(self, gemini_provider):
        """Test that delete_batch method uses genai client correctly."""
        import inspect
        source = inspect.getsource(gemini_provider.delete_batch)
        
        # Verify that we're using genai client for deletion
        assert 'self.genai_client.batches.delete(' in source, "Should use genai client for batch deletion"
        assert 'name=batch_name' in source, "Should pass batch_name as name parameter"
        
        # Verify method signature and return type
        assert 'batch_name: str' in source, "Should have batch_name parameter"
        assert '-> bool:' in source, "Should return boolean"

    def test_batch_management_methods_exist(self, gemini_provider):
        """Test that all batch management methods are implemented."""
        # Verify that all new methods exist
        assert hasattr(gemini_provider, 'list_batches'), "Should have list_batches method"
        assert hasattr(gemini_provider, 'get_batch_by_name'), "Should have get_batch_by_name method"
        assert hasattr(gemini_provider, 'delete_batch'), "Should have delete_batch method"
        
        # Verify they are callable
        assert callable(gemini_provider.list_batches), "list_batches should be callable"
        assert callable(gemini_provider.get_batch_by_name), "get_batch_by_name should be callable"
        assert callable(gemini_provider.delete_batch), "delete_batch should be callable"

    def test_upload_screenshots_early_exit_behavior(self, gemini_provider):
        """Test that screenshot upload methods exit early when no work needed."""
        import inspect
        
        # Test upload_required_screenshots has early exit for empty list
        upload_required_source = inspect.getsource(gemini_provider.upload_required_screenshots)
        assert 'if not required_screenshot_paths:' in upload_required_source, "Should check for empty screenshot paths"
        assert 'no experiments to process' in upload_required_source, "Should mention no experiments in message"
        
        # Test upload_all_screenshots has early exit for no files found
        upload_all_source = inspect.getsource(gemini_provider.upload_all_screenshots)
        assert 'if not screenshot_files:' in upload_all_source, "Should check for empty screenshot files"
        assert 'No screenshot files found' in upload_all_source, "Should mention no files found"

    def test_upload_required_screenshots_empty_list(self, gemini_provider):
        """Test that upload_required_screenshots handles empty list gracefully."""
        import asyncio
        
        # This should not raise an exception and should exit early
        asyncio.run(gemini_provider.upload_required_screenshots([]))
        
        # Verify that no uploads were attempted (no entries in uploaded_screenshots)
        # Since we passed an empty list, the mapping should remain empty
        assert len(gemini_provider._uploaded_screenshots) == 0, "Should not have any uploaded screenshots"

    def test_batch_output_mapping_initialization(self, gemini_provider):
        """Test that batch output mapping is properly initialized."""
        # Verify that the output mapping is initialized
        assert hasattr(gemini_provider, '_batch_output_mappings'), "Should have _batch_output_mappings attribute"
        assert isinstance(gemini_provider._batch_output_mappings, dict), "Should be a dictionary"
        assert len(gemini_provider._batch_output_mappings) == 0, "Should start empty"

    def test_batch_output_mapping_storage(self, gemini_provider):
        """Test that output URI mapping can be stored and retrieved."""
        # Test storing a mapping
        batch_id = "test_batch_123"
        output_uri = "gs://test-bucket/batch_outputs/test_config/"
        
        gemini_provider._batch_output_mappings[batch_id] = output_uri
        
        # Verify it can be retrieved
        assert batch_id in gemini_provider._batch_output_mappings, "Should store batch mapping"
        assert gemini_provider._batch_output_mappings[batch_id] == output_uri, "Should retrieve correct URI"

    def test_has_experiments_to_submit_empty_list(self, gemini_provider):
        """Test has_experiments_to_submit with empty experiment list."""
        assert gemini_provider.has_experiments_to_submit([]) is False, "Should return False for empty list"

    def test_has_experiments_to_submit_valid_experiments(self, gemini_provider, mock_experiment_data):
        """Test has_experiments_to_submit with valid experiments."""
        experiments = [mock_experiment_data]
        assert gemini_provider.has_experiments_to_submit(experiments) is True, "Should return True for valid experiments"

    def test_has_experiments_to_submit_invalid_experiments(self, gemini_provider):
        """Test has_experiments_to_submit with invalid experiments."""
        # Create mock objects without experiment_id
        invalid_experiments = [object(), object()]
        assert gemini_provider.has_experiments_to_submit(invalid_experiments) is False, "Should return False for invalid experiments"


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v"])