"""
Tests for screenshot runtime classes.

This module tests the LocalDatasetRuntime and HFHubDatasetRuntime classes
using mock implementations to avoid actual API calls and file dependencies.
"""

import asyncio
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from typing import List

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from agent.src.typedefs import EngineParams, EngineType
from experiments.config import ExperimentData
from experiments.runners.screenshot_runtime import BaseScreenshotRuntime
from experiments.runners.screenshot_runtime.local_dataset import LocalDatasetRuntime
from experiments.runners.screenshot_runtime.hf_hub_dataset import HFHubDatasetRuntime
from tests.mocks import MockLMMAgent


@pytest.fixture
def mock_engine_params():
    """Create mock engine parameters for testing."""
    return [
        EngineParams(
            engine_type=EngineType.OPENAI,
            model="gpt-4-vision-preview",
            config_name="test_openai",
            api_key="test_key_openai",
            temperature=0.1,
            max_new_tokens=1000
        ),
        EngineParams(
            engine_type=EngineType.ANTHROPIC,
            model="claude-3-sonnet-20240229",
            config_name="test_anthropic",
            api_key="test_key_anthropic",
            temperature=0.1,
            max_new_tokens=1000
        )
    ]


@pytest.fixture
def mock_experiment_data():
    """Create mock experiment data for testing."""
    mock_df = Mock()
    mock_df.copy.return_value = mock_df
    
    return ExperimentData(
        query="wireless mouse",
        experiment_label="test_experiment",
        experiment_number=1,
        prompt_template="Find a wireless mouse and add it to cart",
        experiment_df=mock_df,
        screenshot=None
    )


@pytest.fixture
def mock_experiments_list(mock_experiment_data):
    """Create a list of mock experiments."""
    experiments = []
    for i in range(3):
        exp = ExperimentData(
            query=f"test_query_{i}",
            experiment_label=f"test_label_{i}",
            experiment_number=i,
            prompt_template=f"Test prompt {i}",
            experiment_df=Mock(),
            screenshot=None
        )
        experiments.append(exp)
    return experiments


class TestBaseScreenshotRuntime:
    """Test the BaseScreenshotRuntime abstract base class."""
    
    def test_cannot_instantiate_abstract_class(self, mock_engine_params):
        """Test that the abstract base class cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseScreenshotRuntime(
                dataset_name="test",
                engine_params_list=mock_engine_params
            )
    
    def test_api_key_loading_single_key(self):
        """Test loading a single API key from environment."""
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test_key_1'}):
            keys = BaseScreenshotRuntime.load_api_keys_for_provider(EngineType.OPENAI)
            assert keys == ['test_key_1']
    
    def test_api_key_loading_multiple_keys(self):
        """Test loading multiple API keys from environment."""
        with patch.dict('os.environ', {
            'OPENAI_API_KEY': 'test_key_1',
            'OPENAI_API_KEY_2': 'test_key_2',
            'OPENAI_API_KEY_3': 'test_key_3'
        }):
            keys = BaseScreenshotRuntime.load_api_keys_for_provider(EngineType.OPENAI)
            assert keys == ['test_key_1', 'test_key_2', 'test_key_3']
    
    def test_api_key_loading_no_keys(self):
        """Test behavior when no API keys are found."""
        with patch.dict('os.environ', {}, clear=True):
            keys = BaseScreenshotRuntime.load_api_keys_for_provider(EngineType.OPENAI)
            assert keys == []


class TestLocalDatasetRuntime:
    """Test the LocalDatasetRuntime class."""
    
    @patch('experiments.runners.screenshot_runtime.base.ExperimentWorkerService')
    @patch('experiments.runners.screenshot_runtime.local_dataset.ScreenshotValidationService')
    @patch('experiments.runners.screenshot_runtime.local_dataset.load_experiment_data')
    def test_initialization(self, mock_load_data, mock_validation_service, mock_worker_service, mock_engine_params):
        """Test LocalDatasetRuntime initialization."""
        # Mock the dataset loading
        mock_dataset = Mock()
        mock_load_data.return_value = mock_dataset
        
        # Mock the services
        mock_validation_instance = Mock()
        mock_validation_service.return_value = mock_validation_instance
        mock_worker_instance = Mock()
        mock_worker_service.return_value = mock_worker_instance
        
        runtime = LocalDatasetRuntime(
            local_dataset_path="/test/path/mousepad_dataset.csv",
            engine_params_list=mock_engine_params,
            max_concurrent_per_engine=3,
            experiment_count_limit=10,
            debug_mode=True
        )
        
        # Verify initialization
        assert runtime.dataset_name == "mousepad"
        assert runtime.local_dataset_path == "/test/path/mousepad_dataset.csv"
        assert runtime.experiment_count_limit == 10
        assert runtime.debug_mode == True
        assert runtime.dataset == mock_dataset
        
        # Verify services were initialized
        mock_worker_service.assert_called_once_with(max_concurrent_per_engine=3)
        mock_validation_service.assert_called_once()
        
        # Verify dataset was loaded
        mock_load_data.assert_called_once_with("/test/path/mousepad_dataset.csv")
    
    @patch('experiments.runners.screenshot_runtime.base.ExperimentWorkerService')
    @patch('experiments.runners.screenshot_runtime.local_dataset.ScreenshotValidationService')
    @patch('experiments.runners.screenshot_runtime.local_dataset.experiments_iter')
    @patch('experiments.runners.screenshot_runtime.local_dataset.load_experiment_data')
    def test_experiments_iter_property(self, mock_load_data, mock_experiments_iter, 
                                     mock_validation_service, mock_worker_service, 
                                     mock_engine_params, mock_experiments_list):
        """Test the experiments_iter property."""
        mock_load_data.return_value = Mock()
        mock_experiments_iter.return_value = iter(mock_experiments_list)
        
        runtime = LocalDatasetRuntime(
            local_dataset_path="/test/path/test_dataset.csv",
            engine_params_list=mock_engine_params
        )
        
        experiments = list(runtime.experiments_iter)
        assert len(experiments) == 3
        assert experiments[0].query == "test_query_0"
        
        mock_experiments_iter.assert_called_once_with(runtime.dataset)
    
    @patch('experiments.runners.screenshot_runtime.base.ExperimentWorkerService')
    @patch('experiments.runners.screenshot_runtime.local_dataset.ScreenshotValidationService')
    @patch('experiments.runners.screenshot_runtime.local_dataset.load_experiment_data')
    def test_create_shopping_environment(self, mock_load_data, mock_validation_service, 
                                       mock_worker_service, mock_engine_params, mock_experiment_data):
        """Test creating a shopping environment."""
        mock_load_data.return_value = Mock()
        
        runtime = LocalDatasetRuntime(
            local_dataset_path="/test/path/mousepad_dataset.csv",
            engine_params_list=mock_engine_params,
            remote=True
        )
        
        with patch('experiments.runners.screenshot_runtime.local_dataset.FilesystemShoppingEnvironment') as mock_env:
            mock_env_instance = Mock()
            mock_env.return_value = mock_env_instance
            
            environment = runtime.create_shopping_environment(mock_experiment_data)
            
            # Verify the environment was created with correct parameters
            mock_env.assert_called_once_with(
                screenshots_dir=runtime.screenshots_dir,
                query=mock_experiment_data.query,
                experiment_label=mock_experiment_data.experiment_label,
                experiment_number=mock_experiment_data.experiment_number,
                remote=True
            )
            assert environment == mock_env_instance
    
    @patch('experiments.runners.screenshot_runtime.base.ExperimentWorkerService')
    @patch('experiments.runners.screenshot_runtime.local_dataset.ScreenshotValidationService')
    @patch('experiments.runners.screenshot_runtime.local_dataset.load_experiment_data')
    def test_validate_prerequisites(self, mock_load_data, mock_validation_service, 
                                  mock_worker_service, mock_engine_params):
        """Test prerequisites validation."""
        mock_load_data.return_value = Mock()
        mock_validation_instance = Mock()
        mock_validation_instance.validate_all_screenshots.return_value = True
        mock_validation_service.return_value = mock_validation_instance
        
        runtime = LocalDatasetRuntime(
            local_dataset_path="/test/path/test_dataset.csv",
            engine_params_list=mock_engine_params
        )
        
        result = runtime.validate_prerequisites()
        
        assert result == True
        mock_validation_instance.validate_all_screenshots.assert_called_once_with(
            runtime.dataset,
            "/test/path/test_dataset.csv"
        )
    
    @patch('experiments.runners.screenshot_runtime.base.ExperimentWorkerService')
    @patch('experiments.runners.screenshot_runtime.local_dataset.ScreenshotValidationService')
    @patch('experiments.runners.screenshot_runtime.local_dataset.load_experiment_data')
    @pytest.mark.anyio
    async def test_run_single_experiment(self, mock_load_data, mock_validation_service,
                                       mock_worker_service, mock_engine_params, mock_experiment_data):
        """Test running a single experiment."""
        mock_load_data.return_value = Mock()
        
        runtime = LocalDatasetRuntime(
            local_dataset_path="/test/path/test_dataset.csv",
            engine_params_list=mock_engine_params,
            debug_mode=True
        )
        
        # Mock experiment already exists to test early return path
        with patch.object(mock_experiment_data, 'journey_dir') as mock_journey_dir:
            mock_journey_path = Mock(spec=Path)
            mock_csv_path = Mock(spec=Path)
            mock_csv_path.exists.return_value = True  # Experiment already exists
            mock_journey_path.__truediv__ = Mock(return_value=mock_csv_path)
            mock_journey_dir.return_value = mock_journey_path
            
            # Call the method - should return early due to existing experiment
            await runtime.run_single_experiment(mock_experiment_data, mock_engine_params[0])
            
            # Verify the journey_dir was called to check for existing experiment
            mock_journey_dir.assert_called_once_with(runtime.run_output_dir, mock_engine_params[0])
            mock_csv_path.exists.assert_called_once()


class TestHFHubDatasetRuntime:
    """Test the HFHubDatasetRuntime class."""
    
    @patch('experiments.runners.screenshot_runtime.base.ExperimentWorkerService')
    def test_initialization(self, mock_worker_service, mock_engine_params):
        """Test HFHubDatasetRuntime initialization."""
        mock_worker_instance = Mock()
        mock_worker_service.return_value = mock_worker_instance
        
        runtime = HFHubDatasetRuntime(
            engine_params_list=mock_engine_params,
            hf_dataset_name="test_org/test_dataset",
            subset="mousepad",
            max_concurrent_per_engine=2,
            experiment_count_limit=5,
            debug_mode=False
        )
        
        # Verify initialization
        assert runtime.dataset_name == "test_org_test_dataset_mousepad"
        assert runtime.hf_dataset_name == "test_org/test_dataset"
        assert runtime.subset == "mousepad"
        assert runtime.experiment_count_limit == 5
        assert runtime.debug_mode == False
        
        # Verify services were initialized
        mock_worker_service.assert_called_once_with(max_concurrent_per_engine=2)
    
    @patch('experiments.runners.screenshot_runtime.base.ExperimentWorkerService')
    @patch('experiments.runners.screenshot_runtime.hf_hub_dataset.hf_experiments_iter')
    def test_experiments_iter_property(self, mock_hf_iter, mock_worker_service,
                                     mock_engine_params, mock_experiments_list):
        """Test the experiments_iter property for HF dataset."""
        mock_hf_iter.return_value = iter(mock_experiments_list)
        
        runtime = HFHubDatasetRuntime(
            engine_params_list=mock_engine_params,
            hf_dataset_name="test_org/test_dataset",
            subset="all"
        )
        
        experiments = list(runtime.experiments_iter)
        assert len(experiments) == 3
        assert experiments[1].query == "test_query_1"
        
        mock_hf_iter.assert_called_once_with("test_org/test_dataset", subset="all")
    
    @patch('experiments.runners.screenshot_runtime.base.ExperimentWorkerService')
    def test_validate_prerequisites(self, mock_worker_service, mock_engine_params):
        """Test that HF dataset runtime doesn't need screenshot validation."""
        runtime = HFHubDatasetRuntime(
            engine_params_list=mock_engine_params,
            hf_dataset_name="test_org/test_dataset"
        )
        
        # HF datasets don't need validation
        result = runtime.validate_prerequisites()
        assert result == True
    
    @patch('experiments.runners.screenshot_runtime.base.ExperimentWorkerService')
    def test_create_shopping_environment(self, mock_worker_service, mock_engine_params):
        """Test creating a shopping environment for HF dataset."""
        runtime = HFHubDatasetRuntime(
            engine_params_list=mock_engine_params,
            hf_dataset_name="test_org/test_dataset"
        )
        
        # Create experiment data with a screenshot
        mock_screenshot = Mock()
        experiment_data = ExperimentData(
            query="test_query",
            experiment_label="test_label",
            experiment_number=1,
            prompt_template="Test prompt",
            experiment_df=Mock(),
            screenshot=mock_screenshot
        )
        
        with patch('experiments.runners.screenshot_runtime.hf_hub_dataset.DatasetShoppingEnvironment') as mock_env:
            mock_env_instance = Mock()
            mock_env.return_value = mock_env_instance
            
            environment = runtime.create_shopping_environment(experiment_data)
            
            # Verify the environment was created with the screenshot
            mock_env.assert_called_once_with(screenshot_image=mock_screenshot)
            assert environment == mock_env_instance
    
    @patch('experiments.runners.screenshot_runtime.base.ExperimentWorkerService')
    def test_create_shopping_environment_no_screenshot_raises_error(self, mock_worker_service, mock_engine_params):
        """Test that missing screenshot raises an error."""
        runtime = HFHubDatasetRuntime(
            engine_params_list=mock_engine_params,
            hf_dataset_name="test_org/test_dataset"
        )
        
        # Create experiment data without a screenshot
        experiment_data = ExperimentData(
            query="test_query",
            experiment_label="test_label",
            experiment_number=1,
            prompt_template="Test prompt",
            experiment_df=Mock(),
            screenshot=None  # No screenshot
        )
        
        with pytest.raises(ValueError, match="No screenshot found for experiment"):
            runtime.create_shopping_environment(experiment_data)


class TestScreenshotRuntimeIntegration:
    """Integration tests for screenshot runtime classes."""
    
    @patch('experiments.runners.screenshot_runtime.base.aggregate_run_data')
    @patch('experiments.runners.screenshot_runtime.base.ExperimentWorkerService')
    @patch('experiments.runners.screenshot_runtime.local_dataset.ScreenshotValidationService')
    @patch('experiments.runners.screenshot_runtime.local_dataset.load_experiment_data')
    @pytest.mark.anyio
    async def test_local_dataset_runtime_full_run(self, mock_load_data, mock_validation_service,
                                                 mock_worker_service, mock_aggregate,
                                                 mock_engine_params, mock_experiments_list):
        """Test a full run of LocalDatasetRuntime with mocked services."""
        # Setup mocks
        mock_load_data.return_value = Mock()
        mock_validation_instance = Mock()
        mock_validation_instance.validate_all_screenshots.return_value = True
        mock_validation_service.return_value = mock_validation_instance
        
        mock_worker_instance = AsyncMock()
        mock_worker_service.return_value = mock_worker_instance
        
        runtime = LocalDatasetRuntime(
            local_dataset_path="/test/path/test_dataset.csv",
            engine_params_list=mock_engine_params,
            experiment_count_limit=2
        )
        
        # Mock experiments_iter using patch
        with patch('experiments.runners.screenshot_runtime.local_dataset.experiments_iter') as mock_experiments_iter:
            mock_experiments_iter.return_value = iter(mock_experiments_list[:2])
            await runtime.run()
        
        # Verify validation was called
        mock_validation_instance.validate_all_screenshots.assert_called_once()
        
        # Verify worker service was used
        mock_worker_instance.run_experiments.assert_called_once()
        
        # Verify aggregation was called
        mock_aggregate.assert_called_once()
    
    @patch('experiments.runners.screenshot_runtime.base.aggregate_run_data')
    @patch('experiments.runners.screenshot_runtime.base.ExperimentWorkerService')
    @pytest.mark.anyio
    async def test_hf_hub_runtime_full_run(self, mock_worker_service, mock_aggregate,
                                         mock_engine_params, mock_experiments_list):
        """Test a full run of HFHubDatasetRuntime with mocked services."""
        # Setup mocks
        mock_worker_instance = AsyncMock()
        mock_worker_service.return_value = mock_worker_instance
        
        runtime = HFHubDatasetRuntime(
            engine_params_list=mock_engine_params,
            hf_dataset_name="test_org/test_dataset",
            subset="test",
            experiment_count_limit=1
        )
        
        # Mock experiments_iter using patch
        with patch('experiments.runners.screenshot_runtime.hf_hub_dataset.hf_experiments_iter') as mock_hf_iter:
            mock_hf_iter.return_value = iter(mock_experiments_list[:1])
            await runtime.run()
        
        # Verify worker service was used
        mock_worker_instance.run_experiments.assert_called_once()
        
        # Verify aggregation was called
        mock_aggregate.assert_called_once()


# Test utilities for mock agent
class TestMockLMMAgent:
    """Test the MockLMMAgent implementation."""
    
    def test_mock_agent_initialization(self):
        """Test mock agent initializes correctly."""
        mock_agent = MockLMMAgent(
            system_prompt="Test system prompt",
            tools=[]
        )
        
        assert mock_agent.system_prompt == "Test system prompt"
        assert len(mock_agent.messages) == 1
        assert isinstance(mock_agent.messages[0], SystemMessage)
    
    def test_mock_agent_add_message(self):
        """Test adding messages to mock agent."""
        mock_agent = MockLMMAgent()
        
        mock_agent.add_message("Hello, agent!", role="user")
        mock_agent.add_message("Hello, user!", role="assistant")
        
        assert len(mock_agent.messages) == 3  # System + 2 messages
        assert isinstance(mock_agent.messages[1], HumanMessage)
        assert isinstance(mock_agent.messages[2], AIMessage)
    
    @pytest.mark.anyio
    async def test_mock_agent_response_generation(self):
        """Test mock agent generates responses."""
        mock_agent = MockLMMAgent()
        mock_agent.configure_shopping_response(
            product_name="Test Product",
            product_price=29.99,
            justification="Perfect for testing"
        )
        
        response = await mock_agent.aget_response()
        
        assert isinstance(response, AIMessage)
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0]["name"] == "add_to_cart"
    
    def test_mock_agent_raw_messages(self):
        """Test getting raw messages from mock agent."""
        mock_agent = MockLMMAgent()
        mock_agent.add_message("Test message", role="user")
        
        raw_messages = mock_agent.raw_message_requests()
        
        assert len(raw_messages) == 2  # System + user message
        assert raw_messages[0]["role"] == "system"
        assert raw_messages[1]["role"] == "user"
        assert raw_messages[1]["content"] == "Test message"


class TestBaseScreenshotRuntimeAdditional:
    """Additional tests for BaseScreenshotRuntime functionality."""
    
    def test_distribute_engine_params_with_no_api_keys(self, mock_engine_params):
        """Test engine parameter distribution when no API keys are found."""
        with patch('experiments.runners.screenshot_runtime.base.BaseScreenshotRuntime.load_api_keys_for_provider') as mock_load_keys:
            mock_load_keys.return_value = []
            
            # Create a concrete implementation for testing
            class TestRuntime(BaseScreenshotRuntime):
                @property
                def experiments_iter(self):
                    return iter([])
                
                def create_shopping_environment(self, data):
                    return Mock()
                
                def get_experiments_dataframe(self):
                    return None
                
                def get_dataset_path(self):
                    return None
            
            # Mock the worker service
            with patch('experiments.runners.screenshot_runtime.base.ExperimentWorkerService'):
                runtime = TestRuntime(
                    dataset_name="test",
                    engine_params_list=mock_engine_params
                )
                
                # Should keep original engine params when no API keys found
                assert len(runtime.distributed_engine_params) == len(mock_engine_params)
                assert runtime.distributed_engine_params[0].config_name == mock_engine_params[0].config_name
                assert runtime.distributed_engine_params[1].config_name == mock_engine_params[1].config_name
    
    def test_distribute_engine_params_with_multiple_api_keys(self, mock_engine_params):
        """Test engine parameter distribution with multiple API keys."""
        with patch('experiments.runners.screenshot_runtime.base.BaseScreenshotRuntime.load_api_keys_for_provider') as mock_load_keys:
            # Return different number of keys for different providers
            def mock_keys(engine_type):
                if engine_type == EngineType.OPENAI:
                    return ['key1', 'key2', 'key3']
                elif engine_type == EngineType.ANTHROPIC:
                    return ['anthropic_key1']
                return []
            
            mock_load_keys.side_effect = mock_keys
            
            class TestRuntime(BaseScreenshotRuntime):
                @property
                def experiments_iter(self):
                    return iter([])
                
                def create_shopping_environment(self, data):
                    return Mock()
                
                def get_experiments_dataframe(self):
                    return None
                
                def get_dataset_path(self):
                    return None
            
            with patch('experiments.runners.screenshot_runtime.base.ExperimentWorkerService'):
                runtime = TestRuntime(
                    dataset_name="test",
                    engine_params_list=mock_engine_params
                )
                
                # Should have 3 + 1 = 4 distributed engine params
                assert len(runtime.distributed_engine_params) == 4
                
                # Check OpenAI keys
                openai_configs = [p for p in runtime.distributed_engine_params if p.engine_type == EngineType.OPENAI]
                assert len(openai_configs) == 3
                # Since config_name is a property that generates from engine_type and model,
                # check that we have 3 different instances with the same generated config_name
                # but different API keys
                config_names = [p.config_name for p in openai_configs]
                api_keys = [p.api_key.get_secret_value() if p.api_key else None for p in openai_configs]
                assert all(name == "openai_gpt-4-vision-preview" for name in config_names)
                assert api_keys == ['key1', 'key2', 'key3']
                
                # Check Anthropic keys
                anthropic_configs = [p for p in runtime.distributed_engine_params if p.engine_type == EngineType.ANTHROPIC]
                assert len(anthropic_configs) == 1
                assert anthropic_configs[0].config_name == "anthropic_claude-3-sonnet-20240229"
                assert anthropic_configs[0].api_key.get_secret_value() == 'anthropic_key1'
    
    def test_distribute_engine_params_preserves_custom_attributes(self, mock_engine_params):
        """Test that custom attributes are preserved during distribution."""
        # Add custom attributes to the mock engine params
        mock_engine_params[0].custom_attr = "custom_value"
        mock_engine_params[0].another_attr = 42
        
        with patch('experiments.runners.screenshot_runtime.base.BaseScreenshotRuntime.load_api_keys_for_provider') as mock_load_keys:
            mock_load_keys.return_value = ['key1', 'key2']
            
            class TestRuntime(BaseScreenshotRuntime):
                @property
                def experiments_iter(self):
                    return iter([])
                
                def create_shopping_environment(self, data):
                    return Mock()
                
                def get_experiments_dataframe(self):
                    return None
                
                def get_dataset_path(self):
                    return None
            
            with patch('experiments.runners.screenshot_runtime.base.ExperimentWorkerService'):
                runtime = TestRuntime(
                    dataset_name="test",
                    engine_params_list=mock_engine_params[:1]  # Only test first one
                )
                
                # Should have 2 distributed configs
                assert len(runtime.distributed_engine_params) == 2
                
                # Check that custom attributes were preserved
                for config in runtime.distributed_engine_params:
                    if hasattr(config, 'custom_attr'):
                        assert config.custom_attr == "custom_value"
                    if hasattr(config, 'another_attr'):
                        assert config.another_attr == 42
    
    @patch('experiments.runners.screenshot_runtime.base.aggregate_run_data')
    @patch('experiments.runners.screenshot_runtime.base.ExperimentWorkerService')
    @pytest.mark.anyio
    async def test_run_method_validation_failure(self, mock_worker_service, mock_aggregate):
        """Test run method when validation fails."""
        class TestRuntime(BaseScreenshotRuntime):
            @property
            def experiments_iter(self):
                return iter([])
            
            def create_shopping_environment(self, data):
                return Mock()
            
            def get_experiments_dataframe(self):
                return None
            
            def get_dataset_path(self):
                return None
            
            def validate_prerequisites(self):
                return False  # Force validation failure
        
        mock_engine_param = EngineParams(
            engine_type=EngineType.OPENAI,
            model="gpt-4",
            config_name="test_config",
            api_key="test_key"
        )
        
        runtime = TestRuntime(
            dataset_name="test",
            engine_params_list=[mock_engine_param]
        )
        
        # Should raise RuntimeError when validation fails
        with pytest.raises(RuntimeError, match="Prerequisites validation failed"):
            await runtime.run()
        
        # Worker service should not be called
        mock_worker_service.return_value.run_experiments.assert_not_called()
        mock_aggregate.assert_not_called()
    
    @patch('experiments.runners.screenshot_runtime.base.aggregate_run_data')
    @patch('experiments.runners.screenshot_runtime.base.ExperimentWorkerService')
    @pytest.mark.anyio
    async def test_run_method_worker_service_exception(self, mock_worker_service, mock_aggregate):
        """Test run method when worker service throws exception."""
        mock_worker_instance = AsyncMock()
        mock_worker_instance.run_experiments.side_effect = Exception("Worker service failed")
        mock_worker_service.return_value = mock_worker_instance
        
        class TestRuntime(BaseScreenshotRuntime):
            @property
            def experiments_iter(self):
                return iter([])
            
            def create_shopping_environment(self, data):
                return Mock()
            
            def get_experiments_dataframe(self):
                return None
            
            def get_dataset_path(self):
                return None
        
        mock_engine_param = EngineParams(
            engine_type=EngineType.OPENAI,
            model="gpt-4",
            config_name="test_config",
            api_key="test_key"
        )
        
        runtime = TestRuntime(
            dataset_name="test",
            engine_params_list=[mock_engine_param]
        )
        
        # Should propagate the exception
        with pytest.raises(Exception, match="Worker service failed"):
            await runtime.run()
        
        # Aggregation should not be called
        mock_aggregate.assert_not_called()


class TestLocalDatasetRuntimeAdditional:
    """Additional tests for LocalDatasetRuntime functionality."""
    
    @patch('experiments.runners.screenshot_runtime.local_dataset.load_experiment_data')
    def test_initialization_with_invalid_dataset_path(self, mock_load_data):
        """Test initialization with invalid dataset path."""
        mock_load_data.side_effect = FileNotFoundError("Dataset file not found")
        
        mock_engine_param = EngineParams(
            engine_type=EngineType.OPENAI,
            model="gpt-4",
            config_name="test_config",
            api_key="test_key"
        )
        
        with pytest.raises(FileNotFoundError):
            LocalDatasetRuntime(
                local_dataset_path="/nonexistent/path/dataset.csv",
                engine_params_list=[mock_engine_param]
            )
    
    @patch('experiments.runners.screenshot_runtime.base.ExperimentWorkerService')
    @patch('experiments.runners.screenshot_runtime.local_dataset.ScreenshotValidationService')
    @patch('experiments.runners.screenshot_runtime.local_dataset.load_experiment_data')
    def test_dataset_name_extraction_edge_cases(self, mock_load_data, mock_validation_service, mock_worker_service):
        """Test dataset name extraction from various path formats."""
        mock_load_data.return_value = Mock()
        
        test_cases = [
            ("/path/to/mousepad_dataset.csv", "mousepad"),
            ("/path/to/complex_name_dataset.csv", "complex_name"),
            ("/path/to/simple.csv", "simple"),
            ("/path/to/dataset_with_underscores.csv", "dataset_with_underscores"),
        ]
        
        mock_engine_param = EngineParams(
            engine_type=EngineType.OPENAI,
            model="gpt-4",
            config_name="test_config",
            api_key="test_key"
        )
        
        for path, expected_name in test_cases:
            runtime = LocalDatasetRuntime(
                local_dataset_path=path,
                engine_params_list=[mock_engine_param]
            )
            assert runtime.dataset_name == expected_name
    
    @patch('experiments.runners.screenshot_runtime.base.ExperimentWorkerService')
    @patch('experiments.runners.screenshot_runtime.local_dataset.ScreenshotValidationService')
    @patch('experiments.runners.screenshot_runtime.local_dataset.load_experiment_data')
    def test_validate_prerequisites_service_exception(self, mock_load_data, mock_validation_service, mock_worker_service):
        """Test validation when screenshot validation service throws exception."""
        mock_load_data.return_value = Mock()
        mock_validation_instance = Mock()
        mock_validation_instance.validate_all_screenshots.side_effect = Exception("Validation failed")
        mock_validation_service.return_value = mock_validation_instance
        
        mock_engine_param = EngineParams(
            engine_type=EngineType.OPENAI,
            model="gpt-4",
            config_name="test_config",
            api_key="test_key"
        )
        
        runtime = LocalDatasetRuntime(
            local_dataset_path="/test/path/dataset.csv",
            engine_params_list=[mock_engine_param]
        )
        
        # Should propagate the exception
        with pytest.raises(Exception, match="Validation failed"):
            runtime.validate_prerequisites()


class TestHFHubDatasetRuntimeAdditional:
    """Additional tests for HFHubDatasetRuntime functionality."""
    
    @patch('experiments.runners.screenshot_runtime.base.ExperimentWorkerService')
    def test_dataset_name_generation_edge_cases(self, mock_worker_service):
        """Test dataset name generation with various HF dataset names."""
        test_cases = [
            ("org/dataset", "subset", "org_dataset_subset"),
            ("complex-org/dataset-name", "all", "complex-org_dataset-name_all"),
            ("simple_dataset", "test", "simple_dataset_test"),
            ("org/dataset/config", "subset", "org_dataset_config_subset"),
        ]
        
        mock_engine_param = EngineParams(
            engine_type=EngineType.OPENAI,
            model="gpt-4",
            config_name="test_config",
            api_key="test_key"
        )
        
        for hf_name, subset, expected_name in test_cases:
            runtime = HFHubDatasetRuntime(
                engine_params_list=[mock_engine_param],
                hf_dataset_name=hf_name,
                subset=subset
            )
            assert runtime.dataset_name == expected_name
    
    @patch('experiments.runners.screenshot_runtime.base.ExperimentWorkerService')
    @patch('experiments.runners.screenshot_runtime.hf_hub_dataset.hf_experiments_iter')
    def test_experiments_iter_with_dataset_exception(self, mock_hf_iter, mock_worker_service):
        """Test experiments_iter when HF dataset loading throws exception."""
        mock_hf_iter.side_effect = Exception("HuggingFace dataset loading failed")
        
        mock_engine_param = EngineParams(
            engine_type=EngineType.OPENAI,
            model="gpt-4",
            config_name="test_config",
            api_key="test_key"
        )
        
        runtime = HFHubDatasetRuntime(
            engine_params_list=[mock_engine_param],
            hf_dataset_name="test/dataset"
        )
        
        # Should propagate the exception when iterator is accessed
        with pytest.raises(Exception, match="HuggingFace dataset loading failed"):
            list(runtime.experiments_iter)
    
    @patch('experiments.runners.screenshot_runtime.base.ExperimentWorkerService')
    def test_create_shopping_environment_with_detailed_error(self, mock_worker_service):
        """Test create_shopping_environment with detailed error message."""
        mock_engine_param = EngineParams(
            engine_type=EngineType.OPENAI,
            model="gpt-4",
            config_name="test_config",
            api_key="test_key"
        )
        
        runtime = HFHubDatasetRuntime(
            engine_params_list=[mock_engine_param],
            hf_dataset_name="test/dataset"
        )
        
        experiment_data = ExperimentData(
            query="test_query",
            experiment_label="test_label",
            experiment_number=42,
            prompt_template="Test prompt",
            experiment_df=Mock(),
            screenshot=None  # No screenshot
        )
        
        # Should include experiment details in error message
        with pytest.raises(ValueError) as exc_info:
            runtime.create_shopping_environment(experiment_data)
        
        error_msg = str(exc_info.value)
        assert "test_query" in error_msg
        assert "test_label" in error_msg
        assert "42" in error_msg


class TestAPIKeyLoadingEdgeCases:
    """Test edge cases for API key loading functionality."""
    
    def test_load_api_keys_with_empty_strings(self):
        """Test API key loading when environment variables contain empty strings."""
        with patch.dict('os.environ', {
            'OPENAI_API_KEY': '',
            'OPENAI_API_KEY_2': 'valid_key',
            'OPENAI_API_KEY_3': '   ',  # Whitespace only
        }):
            keys = BaseScreenshotRuntime.load_api_keys_for_provider(EngineType.OPENAI)
            # Empty strings are filtered out, whitespace-only strings are kept
            assert len(keys) == 2
            assert keys == ['valid_key', '   ']
    
    def test_load_api_keys_with_gaps_in_numbering(self):
        """Test API key loading with gaps in numbering."""
        with patch.dict('os.environ', {
            'OPENAI_API_KEY': 'key1',
            'OPENAI_API_KEY_3': 'key3',  # Missing key_2
            'OPENAI_API_KEY_5': 'key5',  # Missing key_4
        }):
            keys = BaseScreenshotRuntime.load_api_keys_for_provider(EngineType.OPENAI)
            # Should stop at first missing number
            assert keys == ['key1']
    
    def test_load_api_keys_unsupported_engine_type(self):
        """Test API key loading for unsupported engine type."""
        # Create a mock engine type that doesn't have env_var_prefix
        mock_engine_type = Mock()
        mock_engine_type.env_var_prefix = None
        
        keys = BaseScreenshotRuntime.load_api_keys_for_provider(mock_engine_type)
        assert keys == []


if __name__ == "__main__":
    # Run tests if this file is executed directly
    pytest.main([__file__, "-v"])