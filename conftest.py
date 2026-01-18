"""Shared pytest fixtures for EngineParams across all tests."""

import pytest

from agent.src.typedefs import (
    AnthropicParams,
    BedrockParams,
    EngineParams,
    EngineType,
    GeminiParams,
    HuggingFaceParams,
    OpenAIParams,
)


@pytest.fixture
def mock_engine_params() -> list[EngineParams]:
    """Provide a list of engine configurations for tests."""
    return [
        EngineParams(
            engine_type=EngineType.OPENAI,
            model="gpt-4",
            display_name="GPT 4",
            api_key="key-openai",
            temperature=0.7,
            max_new_tokens=1000,
        ),
        EngineParams(
            engine_type=EngineType.ANTHROPIC,
            model="claude-3",
            display_name="Claude 3",
            api_key="key-anthropic",
            temperature=0.5,
            max_new_tokens=1000,
        ),
        EngineParams(
            engine_type=EngineType.GEMINI,
            model="gemini-pro",
            display_name="Gemini Pro",
            api_key="key-gemini",
            temperature=0.3,
            max_new_tokens=1000,
        ),
    ]


@pytest.fixture
def mock_openai_params() -> OpenAIParams:
    """Provide OpenAI engine configuration for tests."""
    return OpenAIParams(
        model="gpt-4.1",
        display_name="GPT 4.1",
        api_key="key-openai",
        temperature=0.0,
        max_new_tokens=1000,
    )


@pytest.fixture
def mock_anthropic_params() -> AnthropicParams:
    """Provide Anthropic engine configuration for tests."""
    return AnthropicParams(
        model="claude-3-5-sonnet-20241022",
        display_name="Claude 3.5 Sonnet",
        api_key="key-anthropic",
        temperature=0.0,
        max_new_tokens=1000,
    )


@pytest.fixture
def mock_gemini_params() -> GeminiParams:
    """Provide Gemini engine configuration for tests."""
    return GeminiParams(
        model="gemini-2.0-flash",
        display_name="Gemini 2.0 Flash",
        api_key="key-gemini",
        temperature=0.0,
        max_new_tokens=1000,
    )


@pytest.fixture
def mock_bedrock_params() -> BedrockParams:
    """Provide Bedrock engine configuration for tests."""
    return BedrockParams(
        model="amazon.nova-lite-v1:0",
        display_name="Amazon Nova Lite",
        temperature=0.0,
        max_new_tokens=1000,
        region_name="us-east-1",
    )


@pytest.fixture
def mock_huggingface_params() -> HuggingFaceParams:
    """Provide HuggingFace engine configuration for tests."""
    return HuggingFaceParams(
        display_name="HuggingFace TGI",
        endpoint_url="https://api.example.com/tgi",
        temperature=0.0,
        max_new_tokens=1000,
    )


@pytest.fixture
def dummy_engine_params() -> dict:
    """Provide engine params as dict for tests that use dict-based initialization."""
    return {
        "engine_type": "anthropic",
        "model": "claude-3-sonnet-20240229",
        "display_name": "Claude 3 Sonnet",
        "api_key": "dummy-key-for-testing",
        "temperature": 0.0,
    }
