""" Type definitions used throughout the simulator. """

from enum import StrEnum
from typing import Optional, Any, Dict, List, NewType, Literal

from google.genai.types import ThinkingLevel
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage, ToolCall
from pydantic import BaseModel, Field, field_validator, ConfigDict, SecretStr, model_validator

ChatHistory: List[SystemMessage | HumanMessage | AIMessage | ToolMessage]

# TODO: move to `common` submodule

EngineConfigName = NewType('EngineConfigName', str)


class EngineType(StrEnum):
    """Supported LLM engine types."""
    BEDROCK = "bedrock"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    HUGGINGFACE = "huggingface"

    @property
    def env_var_prefix(self) -> str:
        """ Map engine types to environment variable prefixes """
        match self:
            case EngineType.OPENAI:
                return "OPENAI_API_KEY"
            case EngineType.ANTHROPIC:
                return "ANTHROPIC_API_KEY"
            case EngineType.GEMINI:
                return "GOOGLE_API_KEY"
            case EngineType.HUGGINGFACE:
                return "HUGGINGFACE_API_KEY"
            case EngineType.BEDROCK:
                return "AWS_ACCESS_KEY_ID"
        raise ValueError(f"Unknown engine type: {self}")


class EngineParams(BaseModel):
    """Strongly typed engine parameters for LLM configuration."""
    
    model_config = ConfigDict(
        extra="allow",  # Allow additional fields for provider-specific options
        str_strip_whitespace=True,
    )
    
    engine_type: EngineType = Field(..., description="Type of LLM engine to use")
    model: str = Field(..., description="Model name/identifier")
    api_key: Optional[SecretStr] = Field(None, description="API key for the service")
    temperature: float = Field(0.0, ge=0.0, le=2.0, description="Sampling temperature")
    max_new_tokens: Optional[int] = Field(None, gt=0, description="Maximum tokens to generate")
    
    # Provider-specific fields
    base_url: Optional[str] = Field(None, description="Custom base URL for API (Gemini)")
    endpoint_url: Optional[str] = Field(None, description="Endpoint URL (HuggingFace)")
    thinking: bool = Field(False, description="Enable thinking mode (Anthropic)")
    rate_limit: int = Field(-1, description="Rate limit requests per minute (-1 for no limit)")

    @field_validator("model")
    def validate_model(cls, v):
        """Ensure model is not empty."""
        if not v or not v.strip():
            raise ValueError("Model name cannot be empty")
        return v.strip()
    
    @field_validator("base_url", "endpoint_url")
    def validate_urls(cls, v):
        """Basic URL validation."""
        if v is not None and not v.strip():
            return None
        return v
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for backward compatibility."""
        return self.model_dump(exclude_none=True)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EngineParams":
        """Create EngineParams from dictionary."""
        return cls(**data)

    @property
    def config_name(self) -> EngineConfigName:
        return EngineConfigName(f"{self.engine_type}_{self.model}")


class BedrockParams(EngineParams):
    engine_type: EngineType = Field(EngineType.BEDROCK, frozen=True)

    region_name: Optional[str] = Field(None, description="AWS region name")


class OpenAIParams(EngineParams):
    """OpenAI-specific parameters."""
    
    engine_type: EngineType = Field(EngineType.OPENAI, frozen=True)
    model: str = Field(..., description="OpenAI model name (e.g., gpt-4, gpt-3.5-turbo)")
    
    # OpenAI-specific fields
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0, description="Nucleus sampling parameter")
    frequency_penalty: Optional[float] = Field(None, ge=-2.0, le=2.0, description="Frequency penalty")
    presence_penalty: Optional[float] = Field(None, ge=-2.0, le=2.0, description="Presence penalty")
    stop: Optional[List[str]] = Field(None, description="Stop sequences")
    reasoning_effort: Optional[Literal["minimal"]] = Field(None)


class AnthropicParams(EngineParams):
    """Anthropic-specific parameters."""
    
    engine_type: EngineType = Field(EngineType.ANTHROPIC, frozen=True)
    model: str = Field(..., description="Anthropic model name (e.g., claude-3-sonnet-20240229)")
    
    # Anthropic-specific fields
    thinking: Optional[dict[str, Any]] = Field(None, description="Thinking mode config for Claude")
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0, description="Nucleus sampling parameter")
    top_k: Optional[int] = Field(None, gt=0, description="Top-k sampling parameter")


class GeminiParams(EngineParams):
    """Gemini-specific parameters."""
    
    engine_type: EngineType = Field(EngineType.GEMINI, frozen=True)
    model: str = Field(..., description="Gemini model name (e.g., gemini-pro)")
    base_url: Optional[str] = Field(None, description="Gemini API endpoint URL")
    
    # Gemini-specific fields
    safety_settings: Optional[Dict[str, Any]] = Field(None, description="Safety filter settings")
    thinking_budget: Optional[int] = Field(None, ge=0, description="Thinking budget")
    thinking_level: Optional[ThinkingLevel] = None

    @model_validator(mode="after")
    def validate_thinking_config(self):
        if self.thinking_level is not None and self.thinking_budget is not None:
            raise ValueError(
                "Both thinking_level and thinking_budget cannot be specified together. "
                "See https://ai.google.dev/gemini-api/docs/thinking?utm_source=chatgpt.com#levels-budgets"
            )
        return self


class HuggingFaceParams(EngineParams):
    """HuggingFace-specific parameters."""
    
    engine_type: EngineType = Field(EngineType.HUGGINGFACE, frozen=True)
    model: str = Field("tgi", frozen=True, description="Model identifier (typically 'tgi')")
    endpoint_url: str = Field(..., description="HuggingFace inference endpoint URL")
    
    # HuggingFace-specific fields
    use_cache: bool = Field(True, description="Whether to use caching")
    wait_for_model: bool = Field(False, description="Wait for model to load if not ready")


# ---------------------------------------------------------------- #
#                           Agent Types                            #
# ---------------------------------------------------------------- #

class ShoppingAgentResponse(BaseModel):
    """ Response from the shopping agent. """
    text: str
    tool_call: ToolCall
    tool_response: Optional[str] = None  # stores tool results
