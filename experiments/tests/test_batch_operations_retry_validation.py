#!/usr/bin/env python3
"""
Comprehensive test suite for BatchOperationsService retry logic and metadata validation.

Tests cover:
- Retry logic with backoff for submissions, downloads, and monitoring
- Metadata validation against provider state
- Failed submission handling (not adding to metadata)
- Startup validation and cleanup
- Provider-specific validation (focusing on Gemini)
"""

import asyncio
import json
import os
import pytest
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import pandas as pd

# Import the modules we're testing
from experiments.runners.batch_runtime.services.batch_operations import BatchOperationsService
from experiments.runners.batch_runtime.services.experiment_tracking import ExperimentTrackingService, ExperimentStatus
from experiments.runners.batch_runtime.services.file_operations import FileOperationsService
from experiments.runners.batch_runtime.providers.base import BaseBatchProvider
from experiments.config import ExperimentData
from agent.src.typedefs import EngineParams, EngineType


class MockExperimentData:
    """Mock ExperimentData for testing."""
    
    def __init__(self, query: str = "mousepad", experiment_label: str = "baseline", 
                 experiment_number: int = 1):
        self.query = query
        self.experiment_label = experiment_label
        self.experiment_number = experiment_number
        self.experiment_id = f"{query}_{experiment_label}_{experiment_number}"
        self.prompt_template = "Test prompt"
        self.experiment_df = pd.DataFrame({
            'product_title': ['SteelSeries QcK Gaming Mouse Pad', 'Corsair MM300'],
            'price': [12.99, 24.99],
            'rating': [4.2, 4.8]
        })
    
    def model_output_dir(self, base_path, engine_params):
        """Mock model output directory method."""
        return base_path / f"{engine_params.config_name}_output"


class MockBatchProvider(BaseBatchProvider):
    """Mock batch provider for testing."""
    
    def __init__(self, config_name: str, should_fail: bool = False, fail_validation: bool = False):
        # Create minimal config
        from experiments.runners.batch_runtime.providers.base import BatchProviderConfig
        config = BatchProviderConfig(api_key="test_key")
        
        # Create mock file ops
        file_ops = Mock()
        
        super().__init__(config, file_ops)
        self.config_name = config_name
        self.should_fail = should_fail
        self.fail_validation = fail_validation
        self.submission_count = 0
        self.download_count = 0
        self.monitor_count = 0
        self.submitted_batches = []
    
    async def upload_and_submit_batches(self, batch_requests, config_name, run_output_dir, submission_context):
        """Mock batch submission using the base class implementation."""
        # Use the base class implementation which will call our _submit_chunk
        return await super().upload_and_submit_batches(batch_requests, config_name, run_output_dir, submission_context)
    
    async def monitor_batches(self, batch_ids: List[str]) -> Dict[str, str]:
        """Mock batch monitoring."""
        self.monitor_count += 1
        
        if self.should_fail and self.monitor_count <= 2:  # Fail first 2 attempts
            raise Exception(f"Simulated monitoring failure {self.monitor_count}")
        
        # Return status for each batch
        return {batch_id: "in_progress" if "invalid" in batch_id else "completed" for batch_id in batch_ids}
    
    async def download_batch_results(self, batch_id: str, output_path: Path) -> Optional[str]:
        """Mock batch download."""
        self.download_count += 1
        
        if self.should_fail and self.download_count <= 2:  # Fail first 2 attempts
            raise Exception(f"Simulated download failure {self.download_count}")
        
        # Create mock results file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text('{"test": "result"}')
        return str(output_path)
    
    async def _get_batch_status(self, batch_id: str) -> str:
        """Mock individual batch status check (Gemini-style)."""
        if self.fail_validation:
            raise Exception("Validation check failed")
        
        # Return error for "invalid" batch IDs, success for others
        if "invalid" in batch_id:
            return "error"
        return "completed"
    
    def create_batch_request(self, data, engine_params, raw_messages, custom_id, tools):
        """Mock batch request creation."""
        return {"custom_id": custom_id, "test": "request"}
    
    def parse_tool_calls_from_response(self, response_body):
        """Mock tool call parsing."""
        return [{"function": {"name": "add_to_cart", "arguments": '{"test": true}'}}]
    
    def is_response_successful(self, result):
        """Mock response success check."""
        return "error" not in result
    
    def get_error_message(self, result):
        """Mock error message extraction."""
        return result.get("error", "Unknown error")
    
    def get_response_body_from_result(self, result):
        """Mock response body extraction."""
        return result
    
    def extract_response_content(self, response_body):
        """Mock response content extraction."""
        return "Test response content"
    
    def get_request_chunks(self, requests: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Mock request chunking - split into chunks of 10."""
        chunks = []
        for i in range(0, len(requests), 10):
            chunks.append(requests[i:i+10])
        return chunks
    
    async def _submit_chunk(self, chunk: List[Dict[str, Any]], config_name: str,
                           chunk_index: int, submission_context: Optional[Dict[str, Any]]) -> Optional[str]:
        """Mock chunk submission."""
        if self.should_fail and self.submission_count < 2:  # Fail first 2 attempts
            self.submission_count += 1
            raise Exception(f"Simulated chunk submission failure {self.submission_count}")
        
        self.submission_count += 1
        # Success case
        batch_id = f"batch_{chunk_index}_{self.submission_count}"
        self.submitted_batches.append(batch_id)
        return batch_id


@pytest.fixture
def temp_dir():
    """Create temporary directory for testing."""
    with tempfile.TemporaryDirectory() as temp_path:
        yield Path(temp_path)


@pytest.fixture
def mock_engine_params():
    """Create mock engine parameters."""
    return EngineParams(
        engine_type=EngineType.GEMINI,
        model='gemini-2.0-flash-001',
        temperature=0.7,
        max_tokens=1000
        # config_name is automatically generated as engine_type_model
    )


@pytest.fixture
def mock_experiments_df():
    """Create mock experiments dataframe."""
    return pd.DataFrame({
        'query': ['mousepad', 'mousepad', 'toothpaste'],
        'experiment_label': ['baseline', 'baseline', 'baseline'],
        'experiment_number': [1, 2, 1],
        'product_title': ['SteelSeries QcK', 'Corsair MM300', 'Crest Toothpaste'],
        'price': [12.99, 24.99, 3.99],
        'assigned_position': [1, 2, 1]  # Add this required column
    })


@pytest.fixture
def batch_service_setup(temp_dir, mock_experiments_df):
    """Set up batch operations service with dependencies."""
    # Set up environment variables for testing
    os.environ['GCS_BUCKET_NAME'] = 'test-bucket'
    os.environ['GOOGLE_CLOUD_PROJECT'] = 'test-project'
    
    # Create services
    file_ops = FileOperationsService(temp_dir)
    tracking_service = ExperimentTrackingService(temp_dir)
    
    # Create providers - key must match engine_type_model format
    providers = {
        'gemini_gemini-2.0-flash-001': MockBatchProvider('gemini_gemini-2.0-flash-001'),
        'failing_provider': MockBatchProvider('failing_provider', should_fail=True),
        'validation_failing': MockBatchProvider('validation_failing', fail_validation=True)
    }
    
    # Create batch service
    batch_service = BatchOperationsService(
        providers=providers,
        tracking_service=tracking_service,
        file_ops=file_ops,
        screenshots_dir=temp_dir / "screenshots"
    )
    
    return {
        'batch_service': batch_service,
        'tracking_service': tracking_service,
        'file_ops': file_ops,
        'providers': providers,
        'temp_dir': temp_dir,
        'experiments_df': mock_experiments_df
    }


class TestBatchOperationsRetryLogic:
    """Test retry logic with backoff functionality."""
    
    @pytest.mark.asyncio
    async def test_submit_with_retry_success_first_attempt(self, batch_service_setup, mock_engine_params):
        """Test successful submission on first attempt."""
        setup = batch_service_setup
        batch_service = setup['batch_service']
        provider = setup['providers']['gemini_gemini-2.0-flash-001']
        
        # Test submission
        result = await batch_service._submit_with_retry(
            provider, [{"test": "request"}], "gemini_gemini-2.0-flash-001", setup['temp_dir'], {}
        )
        
        assert result is not None
        assert len(result) == 1
        assert provider.submission_count == 1
    
    @pytest.mark.asyncio
    async def test_submit_with_retry_success_after_failures(self, batch_service_setup, mock_engine_params):
        """Test successful submission after initial failures."""
        setup = batch_service_setup
        batch_service = setup['batch_service']
        
        # Create a simple mock provider that fails then succeeds
        class FailingProvider:
            def __init__(self):
                self.attempt = 0
                
            async def upload_and_submit_batches(self, requests, config_name, output_dir, context):
                self.attempt += 1
                if self.attempt <= 2:
                    raise Exception(f"Simulated failure {self.attempt}")
                return [f"batch_success_{self.attempt}"]
        
        provider = FailingProvider()
        
        # Test submission with retries
        result = await batch_service._submit_with_retry(
            provider, [{"test": "request"}], "failing_provider", setup['temp_dir'], {}
        )
        
        assert result is not None
        assert len(result) == 1
        assert "batch_success_3" in result
        assert provider.attempt == 3  # Failed twice, succeeded on third
    
    @pytest.mark.asyncio
    async def test_download_with_retry_success_after_failures(self, batch_service_setup):
        """Test successful download after initial failures."""
        setup = batch_service_setup
        batch_service = setup['batch_service']
        provider = setup['providers']['failing_provider']
        
        # Test download with retries
        result_file = setup['temp_dir'] / "test_results.jsonl"
        result = await batch_service._download_with_retry(provider, "test_batch", result_file)
        
        assert result is not None
        assert result_file.exists()
        assert provider.download_count == 3  # Failed twice, succeeded on third
    
    @pytest.mark.asyncio
    async def test_monitor_with_retry_success_after_failures(self, batch_service_setup):
        """Test successful monitoring after initial failures."""
        setup = batch_service_setup
        batch_service = setup['batch_service']
        provider = setup['providers']['failing_provider']
        
        # Test monitoring with retries
        result = await batch_service._monitor_with_retry(provider, ["test_batch_1", "test_batch_2"])
        
        assert result is not None
        assert len(result) == 2
        assert provider.monitor_count == 3  # Failed twice, succeeded on third


class TestMetadataValidation:
    """Test metadata validation against providers."""
    
    @pytest.mark.asyncio
    async def test_validate_submitted_batches_all_valid(self, batch_service_setup):
        """Test validation when all submitted batches are valid."""
        setup = batch_service_setup
        batch_service = setup['batch_service']
        provider = setup['providers']['gemini_gemini-2.0-flash-001']
        
        # Submit some valid batches
        batch_ids = ["valid_batch_1", "valid_batch_2"]
        
        # Test validation
        valid_batches = await batch_service._validate_submitted_batches(
            provider, batch_ids, "gemini_gemini-2.0-flash-001"
        )
        
        assert len(valid_batches) == 2
        assert "valid_batch_1" in valid_batches
        assert "valid_batch_2" in valid_batches
    
    @pytest.mark.asyncio
    async def test_validate_submitted_batches_some_invalid(self, batch_service_setup):
        """Test validation when some submitted batches are invalid."""
        setup = batch_service_setup
        batch_service = setup['batch_service']
        provider = setup['providers']['gemini_gemini-2.0-flash-001']
        
        # Mix of valid and invalid batches
        batch_ids = ["valid_batch_1", "invalid_batch_1", "valid_batch_2"]
        
        # Test validation
        valid_batches = await batch_service._validate_submitted_batches(
            provider, batch_ids, "gemini_gemini-2.0-flash-001"
        )
        
        assert len(valid_batches) == 2
        assert "valid_batch_1" in valid_batches
        assert "valid_batch_2" in valid_batches
        assert "invalid_batch_1" not in valid_batches
    
    @pytest.mark.asyncio
    async def test_validate_submitted_batches_validation_error(self, batch_service_setup):
        """Test validation when provider validation fails."""
        setup = batch_service_setup
        batch_service = setup['batch_service']
        provider = setup['providers']['validation_failing']
        
        # Test validation with failing provider
        valid_batches = await batch_service._validate_submitted_batches(
            provider, ["test_batch"], "validation_failing"
        )
        
        # Should return empty list when validation fails
        assert len(valid_batches) == 0


class TestMetadataConsistency:
    """Test that failed submissions don't corrupt metadata."""
    
    @pytest.mark.asyncio
    async def test_failed_submission_no_metadata_update(self, batch_service_setup, mock_engine_params):
        """Test that failed submissions don't update metadata."""
        setup = batch_service_setup
        batch_service = setup['batch_service']
        tracking_service = setup['tracking_service']
        
        # Create experiment data
        experiment_data = MockExperimentData()
        
        # Mock the _submit_with_retry to return None (failure)
        with patch.object(batch_service, '_submit_with_retry', return_value=None):
            # Attempt submission
            result = await batch_service._submit_batches_for_engine(
                mock_engine_params, setup['experiments_df'], setup['temp_dir'], False
            )
            
            assert result == [] or result is None  # Either empty list or None indicates failure
        
        # Check that no metadata was written
        metadata_file = tracking_service.mapping_file
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            assert 'gemini_gemini-2.0-flash-001' not in metadata or not metadata.get('gemini_gemini-2.0-flash-001', {}).get('batches', {})
    
    @pytest.mark.asyncio
    async def test_successful_submission_updates_metadata(self, batch_service_setup, mock_engine_params):
        """Test that successful submissions update metadata."""
        setup = batch_service_setup
        batch_service = setup['batch_service']
        tracking_service = setup['tracking_service']
        
        # Mock experiments_iter to return mock experiment data
        with patch('experiments.runners.batch_runtime.services.batch_operations.experiments_iter') as mock_iter:
            mock_experiments = [MockExperimentData()]
            mock_iter.return_value = mock_experiments
            
            # Test successful submission
            result = await batch_service._submit_batches_for_engine(
                mock_engine_params, setup['experiments_df'], setup['temp_dir'], False
            )
            
            assert len(result) > 0
            
            # Check that metadata was written
            metadata_file = tracking_service.mapping_file
            assert metadata_file.exists()
            
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            assert 'gemini_gemini-2.0-flash-001' in metadata
            assert len(metadata['gemini_gemini-2.0-flash-001']['batches']) > 0


class TestComprehensiveValidation:
    """Test comprehensive metadata validation functionality."""
    
    @pytest.mark.asyncio
    async def test_validate_and_rectify_metadata_no_metadata(self, batch_service_setup):
        """Test validation when no metadata exists."""
        setup = batch_service_setup
        batch_service = setup['batch_service']
        
        # Test validation with no existing metadata
        result = await batch_service.validate_and_rectify_metadata("gemini_gemini-2.0-flash-001", auto_cleanup=True)
        
        assert 'summary' in result
        assert result['summary']['total_batches'] == 0
        assert result['summary']['validation_success_rate'] == 100.0
        assert result['cleanup_performed'] is False
    
    @pytest.mark.asyncio
    async def test_validate_and_rectify_metadata_with_invalid_batches(self, batch_service_setup):
        """Test validation and cleanup with invalid batches."""
        setup = batch_service_setup
        batch_service = setup['batch_service']
        tracking_service = setup['tracking_service']
        
        # Create some metadata with invalid batches
        metadata = {
            'gemini_gemini-2.0-flash-001': {
                'batches': {
                    'valid_batch_1': {
                        'experiment_ids': ['exp_1', 'exp_2'],
                        'submitted_at': 1234567890,
                        'status': 'submitted'
                    },
                    'invalid_batch_1': {
                        'experiment_ids': ['exp_3', 'exp_4'],
                        'submitted_at': 1234567891,
                        'status': 'submitted'
                    }
                },
                'experiments': {
                    'exp_1': {'batch_id': 'valid_batch_1', 'submitted_at': 1234567890},
                    'exp_2': {'batch_id': 'valid_batch_1', 'submitted_at': 1234567890},
                    'exp_3': {'batch_id': 'invalid_batch_1', 'submitted_at': 1234567891},
                    'exp_4': {'batch_id': 'invalid_batch_1', 'submitted_at': 1234567891}
                }
            }
        }
        
        # Write metadata
        tracking_service.mapping_file.parent.mkdir(parents=True, exist_ok=True)
        with open(tracking_service.mapping_file, 'w') as f:
            json.dump(metadata, f)
        
        # Test validation with cleanup
        result = await batch_service.validate_and_rectify_metadata("gemini_gemini-2.0-flash-001", auto_cleanup=True)
        
        assert 'summary' in result
        assert result['summary']['total_batches'] == 2
        assert result['summary']['valid_batches'] == 1
        assert result['summary']['invalid_batches'] == 1
        assert result['summary']['orphaned_experiments'] == 2
        assert result['cleanup_performed'] is True
        
        # Verify cleanup happened
        with open(tracking_service.mapping_file, 'r') as f:
            cleaned_metadata = json.load(f)
        
        # Should only have valid batch left
        assert len(cleaned_metadata['gemini_gemini-2.0-flash-001']['batches']) == 1
        assert 'valid_batch_1' in cleaned_metadata['gemini_gemini-2.0-flash-001']['batches']
        assert 'invalid_batch_1' not in cleaned_metadata['gemini_gemini-2.0-flash-001']['batches']
        
        # Should only have experiments from valid batch
        assert len(cleaned_metadata['gemini_gemini-2.0-flash-001']['experiments']) == 2
        assert 'exp_1' in cleaned_metadata['gemini_gemini-2.0-flash-001']['experiments']
        assert 'exp_2' in cleaned_metadata['gemini_gemini-2.0-flash-001']['experiments']
        assert 'exp_3' not in cleaned_metadata['gemini_gemini-2.0-flash-001']['experiments']
        assert 'exp_4' not in cleaned_metadata['gemini_gemini-2.0-flash-001']['experiments']
    
    @pytest.mark.asyncio
    async def test_validate_and_rectify_metadata_provider_not_found(self, batch_service_setup):
        """Test validation with non-existent provider."""
        setup = batch_service_setup
        batch_service = setup['batch_service']
        
        # Test validation with non-existent provider
        result = await batch_service.validate_and_rectify_metadata("nonexistent_provider")
        
        assert 'error' in result
        assert 'No provider found' in result['error']


class TestProviderSpecificValidation:
    """Test provider-specific validation, focusing on Gemini."""
    
    @pytest.mark.asyncio
    async def test_gemini_provider_validation_uses_get_batch_status(self, batch_service_setup):
        """Test that Gemini provider uses _get_batch_status for validation."""
        setup = batch_service_setup
        batch_service = setup['batch_service']
        provider = setup['providers']['gemini_gemini-2.0-flash-001']
        
        # Verify provider has _get_batch_status method
        assert hasattr(provider, '_get_batch_status')
        
        # Test validation
        batch_ids = ["test_batch_1", "test_batch_2"]
        valid_batches = await batch_service._validate_submitted_batches(
            provider, batch_ids, "gemini_gemini-2.0-flash-001"
        )
        
        assert len(valid_batches) == 2
    
    @pytest.mark.asyncio
    async def test_fallback_to_monitor_batches_for_other_providers(self, batch_service_setup):
        """Test fallback to monitor_batches for providers without _get_batch_status."""
        setup = batch_service_setup
        batch_service = setup['batch_service']
        
        # Create a mock provider that doesn't have _get_batch_status by using a simple class
        class SimpleProvider:
            async def monitor_batches(self, batch_ids):
                return {"test_batch_1": "completed", "test_batch_2": "completed"}
        
        provider = SimpleProvider()
        
        # Test validation falls back to monitor_batches
        batch_ids = ["test_batch_1", "test_batch_2"]
        valid_batches = await batch_service._validate_submitted_batches(
            provider, batch_ids, "test_provider"
        )
        
        assert len(valid_batches) == 2


class TestIntegrationScenarios:
    """Test complete integration scenarios."""
    
    @pytest.mark.asyncio
    async def test_full_submission_workflow_with_retries(self, batch_service_setup, mock_engine_params):
        """Test complete submission workflow with retries and validation."""
        setup = batch_service_setup
        batch_service = setup['batch_service']
        
        # Mock experiments_iter to return mock experiment data
        with patch('experiments.runners.batch_runtime.services.batch_operations.experiments_iter') as mock_iter:
            mock_experiments = [MockExperimentData()]
            mock_iter.return_value = mock_experiments
            
            # Test full submission workflow
            result = await batch_service.submit_all_batches(
                [mock_engine_params], setup['experiments_df'], setup['temp_dir'], False
            )
            
            assert 'gemini_gemini-2.0-flash-001' in result
            assert len(result['gemini_gemini-2.0-flash-001']) > 0
            
            # Verify metadata was created
            tracking_service = setup['tracking_service']
            assert tracking_service.mapping_file.exists()
    
    @pytest.mark.asyncio
    async def test_startup_validation_scenario(self, batch_service_setup):
        """Test startup validation scenario."""
        setup = batch_service_setup
        batch_service = setup['batch_service']
        tracking_service = setup['tracking_service']
        
        # Create some metadata first
        metadata = {
            'gemini_gemini-2.0-flash-001': {
                'batches': {
                    'valid_batch': {
                        'experiment_ids': ['exp_1'],
                        'submitted_at': 1234567890,
                        'status': 'submitted'
                    }
                },
                'experiments': {
                    'exp_1': {'batch_id': 'valid_batch', 'submitted_at': 1234567890}
                }
            }
        }
        
        tracking_service.mapping_file.parent.mkdir(parents=True, exist_ok=True)
        with open(tracking_service.mapping_file, 'w') as f:
            json.dump(metadata, f)
        
        # Test validation
        result = await batch_service.validate_and_rectify_metadata("gemini_gemini-2.0-flash-001", auto_cleanup=True)
        
        assert result['summary']['validation_success_rate'] == 100.0
        assert result['cleanup_performed'] is False  # No cleanup needed


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v"])