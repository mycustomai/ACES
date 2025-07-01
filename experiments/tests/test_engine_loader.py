"""
Tests for the engine loader functionality.
"""

import pytest
import tempfile
import os
from pathlib import Path
from typing import Dict, Any

from experiments.engine_loader import (
    load_yaml_config,
    load_engine_params_from_yaml,
    load_all_model_engine_params
)
from agent.src.typedefs import (
    EngineParams,
    EngineType,
    OpenAIParams,
    AnthropicParams,
    GeminiParams,
    HuggingFaceParams
)


class TestEngineLoader:
    """Test suite for engine loader functionality."""

    def create_temp_yaml_file(self, content: str) -> str:
        """Helper to create temporary YAML files."""
        fd, path = tempfile.mkstemp(suffix='.yaml')
        with os.fdopen(fd, 'w') as f:
            f.write(content)
        return path

    def test_load_yaml_config_basic(self):
        """Test basic YAML config loading."""
        yaml_content = """
engine_type: openai
model: gpt-4
temperature: 0.5
"""
        temp_file = self.create_temp_yaml_file(yaml_content)
        try:
            config = load_yaml_config(temp_file)
            assert config['engine_type'] == 'openai'
            assert config['model'] == 'gpt-4'
            assert config['temperature'] == 0.5
        finally:
            os.unlink(temp_file)

    def test_load_openai_params(self):
        """Test loading OpenAI-specific parameters."""
        yaml_content = """
engine_type: openai
model: gpt-4
temperature: 0.7
max_new_tokens: 1000
top_p: 0.9
frequency_penalty: 0.1
presence_penalty: 0.2
stop: ["END", "STOP"]
"""
        temp_file = self.create_temp_yaml_file(yaml_content)
        try:
            params = load_engine_params_from_yaml(temp_file)
            
            # Should return OpenAIParams instance
            assert isinstance(params, OpenAIParams)
            assert params.engine_type == EngineType.OPENAI
            assert params.model == 'gpt-4'
            assert params.temperature == 0.7
            assert params.max_new_tokens == 1000
            assert params.top_p == 0.9
            assert params.frequency_penalty == 0.1
            assert params.presence_penalty == 0.2
            assert params.stop == ["END", "STOP"]
        finally:
            os.unlink(temp_file)

    def test_load_anthropic_params(self):
        """Test loading Anthropic-specific parameters."""
        yaml_content = """
engine_type: anthropic
model: claude-sonnet-4-20250514
temperature: 1.0
thinking: true
top_p: 0.8
top_k: 50
"""
        temp_file = self.create_temp_yaml_file(yaml_content)
        try:
            params = load_engine_params_from_yaml(temp_file)
            
            # Should return AnthropicParams instance
            assert isinstance(params, AnthropicParams)
            assert params.engine_type == EngineType.ANTHROPIC
            assert params.model == 'claude-sonnet-4-20250514'
            assert params.temperature == 1.0
            assert params.thinking == True
            assert params.top_p == 0.8
            assert params.top_k == 50
        finally:
            os.unlink(temp_file)

    def test_load_gemini_params(self):
        """Test loading Gemini-specific parameters."""
        yaml_content = """
engine_type: gemini
model: gemini-2.5-flash-preview-05-20
temperature: 1.0
base_url: https://generativelanguage.googleapis.com/v1beta/
thinking_budget: 1000
safety_settings:
  harassment: BLOCK_NONE
  hate_speech: BLOCK_NONE
"""
        temp_file = self.create_temp_yaml_file(yaml_content)
        try:
            params = load_engine_params_from_yaml(temp_file)
            
            # Should return GeminiParams instance
            assert isinstance(params, GeminiParams)
            assert params.engine_type == EngineType.GEMINI
            assert params.model == 'gemini-2.5-flash-preview-05-20'
            assert params.temperature == 1.0
            assert params.base_url == 'https://generativelanguage.googleapis.com/v1beta/'
            assert params.thinking_budget == 1000
            assert params.safety_settings == {
                'harassment': 'BLOCK_NONE',
                'hate_speech': 'BLOCK_NONE'
            }
        finally:
            os.unlink(temp_file)

    def test_load_huggingface_params(self):
        """Test loading HuggingFace-specific parameters."""
        yaml_content = """
engine_type: huggingface
model: tgi
endpoint_url: https://api-inference.huggingface.co/models/meta-llama/Llama-2-7b-chat-hf
temperature: 0.8
use_cache: false
wait_for_model: true
"""
        temp_file = self.create_temp_yaml_file(yaml_content)
        try:
            params = load_engine_params_from_yaml(temp_file)
            
            # Should return HuggingFaceParams instance
            assert isinstance(params, HuggingFaceParams)
            assert params.engine_type == EngineType.HUGGINGFACE
            assert params.model == 'tgi'
            assert params.endpoint_url == 'https://api-inference.huggingface.co/models/meta-llama/Llama-2-7b-chat-hf'
            assert params.temperature == 0.8
            assert params.use_cache == False
            assert params.wait_for_model == True
        finally:
            os.unlink(temp_file)

    def test_missing_engine_type_error(self):
        """Test error handling when engine_type is missing."""
        yaml_content = """
model: gpt-4
temperature: 0.5
"""
        temp_file = self.create_temp_yaml_file(yaml_content)
        try:
            with pytest.raises(ValueError, match="Missing 'engine_type'"):
                load_engine_params_from_yaml(temp_file)
        finally:
            os.unlink(temp_file)

    def test_invalid_engine_type_error(self):
        """Test error handling for invalid engine_type."""
        yaml_content = """
engine_type: invalid_engine
model: gpt-4
temperature: 0.5
"""
        temp_file = self.create_temp_yaml_file(yaml_content)
        try:
            with pytest.raises(ValueError, match="Invalid engine_type 'invalid_engine'"):
                load_engine_params_from_yaml(temp_file)
        finally:
            os.unlink(temp_file)

    def test_provider_specific_validation(self):
        """Test that provider-specific validation is enforced."""
        # Test OpenAI validation - invalid frequency_penalty
        yaml_content = """
engine_type: openai
model: gpt-4
temperature: 0.5
frequency_penalty: 5.0
"""
        temp_file = self.create_temp_yaml_file(yaml_content)
        try:
            with pytest.raises(ValueError):
                load_engine_params_from_yaml(temp_file)
        finally:
            os.unlink(temp_file)

        # Test Gemini validation - invalid thinking_budget
        yaml_content = """
engine_type: gemini
model: gemini-pro
temperature: 0.5
thinking_budget: -100
"""
        temp_file = self.create_temp_yaml_file(yaml_content)
        try:
            with pytest.raises(ValueError):
                load_engine_params_from_yaml(temp_file)
        finally:
            os.unlink(temp_file)

    def test_load_all_model_engine_params(self):
        """Test loading all model configs from a directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create multiple config files
            configs = [
                ("openai_gpt4.yaml", """
engine_type: openai
model: gpt-4
temperature: 0.5
"""),
                ("anthropic_claude.yaml", """
engine_type: anthropic
model: claude-3-sonnet
temperature: 0.7
"""),
                ("_disabled_config.yaml", """
engine_type: openai
model: gpt-3.5-turbo
temperature: 0.3
"""),  # Should be skipped (starts with _)
                ("gemini_flash.yaml", """
engine_type: gemini
model: gemini-1.5-flash
temperature: 1.0
""")
            ]
            
            for filename, content in configs:
                with open(os.path.join(temp_dir, filename), 'w') as f:
                    f.write(content)
            
            # Load all configs
            all_params = load_all_model_engine_params(temp_dir)
            
            # Should load 3 configs (excluding the _disabled one)
            assert len(all_params) == 3
            
            # Check types
            types_found = [type(p) for p in all_params]
            assert OpenAIParams in types_found
            assert AnthropicParams in types_found
            assert GeminiParams in types_found

    def test_load_all_with_include_filter(self):
        """Test loading configs with include filter."""
        with tempfile.TemporaryDirectory() as temp_dir:
            configs = [
                ("openai_gpt4.yaml", """
engine_type: openai
model: gpt-4
temperature: 0.5
"""),
                ("anthropic_claude.yaml", """
engine_type: anthropic
model: claude-3-sonnet
temperature: 0.7
""")
            ]
            
            for filename, content in configs:
                with open(os.path.join(temp_dir, filename), 'w') as f:
                    f.write(content)
            
            # Only include openai config
            filtered_params = load_all_model_engine_params(temp_dir, include=["openai_gpt4"])
            
            assert len(filtered_params) == 1
            assert isinstance(filtered_params[0], OpenAIParams)
            assert filtered_params[0].model == "gpt-4"

    def test_load_all_with_exclude_filter(self):
        """Test loading configs with exclude filter."""
        with tempfile.TemporaryDirectory() as temp_dir:
            configs = [
                ("openai_gpt4.yaml", """
engine_type: openai
model: gpt-4
temperature: 0.5
"""),
                ("anthropic_claude.yaml", """
engine_type: anthropic
model: claude-3-sonnet
temperature: 0.7
""")
            ]
            
            for filename, content in configs:
                with open(os.path.join(temp_dir, filename), 'w') as f:
                    f.write(content)
            
            # Exclude openai config
            filtered_params = load_all_model_engine_params(temp_dir, exclude=["openai_gpt4"])
            
            assert len(filtered_params) == 1
            assert isinstance(filtered_params[0], AnthropicParams)
            assert filtered_params[0].model == "claude-3-sonnet"

    def test_load_real_config_files(self):
        """Test loading actual config files from the project."""
        # Test with real config files if they exist
        config_dir = "config/models"
        if os.path.exists(config_dir):
            configs = load_all_model_engine_params(config_dir)
            
            # Should have loaded some configs
            assert len(configs) > 0
            
            # All should be valid EngineParams instances
            for config in configs:
                assert isinstance(config, EngineParams)
                assert config.engine_type in [
                    EngineType.OPENAI, 
                    EngineType.ANTHROPIC, 
                    EngineType.GEMINI, 
                    EngineType.HUGGINGFACE
                ]
                assert config.model is not None
                assert config.temperature >= 0.0