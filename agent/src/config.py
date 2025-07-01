""" Configuration management using pydantic-settings"""

from functools import lru_cache
from typing import Optional

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    """Application configuration loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # API Keys for various LLM providers
    openai_api_key: Optional[SecretStr] = Field(None, description="OpenAI API key")
    anthropic_api_key: Optional[SecretStr] = Field(None, description="Anthropic API key")
    google_api_key: Optional[SecretStr] = Field(None, description="Google API key for Gemini")
    huggingfacehub_api_token: Optional[SecretStr] = Field(None, description="HuggingFace Hub API token")
    
    # Runtime options
    continue_or_exit_wait_delay: int = Field(5, description="Delay in seconds to wait before continuing the experiment after a keyboard interrupt")
    
    # Feature flags
    explicit_product_selection: bool = Field(False, description="Prevent the model from setting the `add_to_cart` flag without a specific product selection")


@lru_cache
def get_config() -> Config:
    """Get application configuration with caching."""
    # noinspection PyArgumentList
    return Config()

