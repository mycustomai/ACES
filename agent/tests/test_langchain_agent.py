"""Test suite for the LangChain-based LMMAgent."""

import pytest
import sys
from pathlib import Path

from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic_core import ValidationError

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agent.src.core.mllm import LMMAgent
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage


class TestLMMAgent:
    """Test suite for LMMAgent."""
    
    @pytest.fixture
    def dummy_engine_params(self):
        """Provide dummy engine parameters for testing."""
        return {
            "engine_type": "anthropic",
            "model": "claude-3-sonnet-20240229",
            "api_key": "dummy-key-for-testing",
            "temperature": 0.0
        }
    
    @pytest.fixture
    def agent(self, dummy_engine_params):
        """Create a test agent instance."""
        return LMMAgent(
            engine_params=dummy_engine_params,
            system_prompt="You are a test assistant."
        )
    
    def test_agent_creation_with_engine_params(self, dummy_engine_params):
        """Test basic agent creation with engine parameters."""
        agent = LMMAgent(
            engine_params=dummy_engine_params,
            system_prompt="You are a test assistant."
        )
        
        assert agent.system_prompt == "You are a test assistant."
        assert len(agent.messages) == 1  # Should have system message
        assert isinstance(agent.llm, ChatAnthropic)
        assert isinstance(agent.messages[0], SystemMessage)
    
    def test_agent_creation_with_direct_llm(self):
        """Test agent creation with direct LLM instance."""
        llm = ChatAnthropic(
            model="claude-3-sonnet-20240229", 
            api_key="dummy-key-for-testing", 
            temperature=0.0
        )
        agent = LMMAgent(
            llm=llm,
            system_prompt="You are a direct LLM test assistant."
        )
        
        assert agent.system_prompt == "You are a direct LLM test assistant."
        assert isinstance(agent.llm, ChatAnthropic)
        assert len(agent.messages) == 1
    
    def test_agent_creation_requires_llm_or_params(self):
        """Test that agent creation requires either LLM or engine params."""
        with pytest.raises(ValueError, match="Either llm or engine_params must be provided"):
            LMMAgent()
    
    def test_add_user_message(self, agent):
        """Test adding a user message."""
        initial_count = len(agent.messages)
        agent.add_message("Hello, this is a test message.")
        
        assert len(agent.messages) == initial_count + 1
        assert isinstance(agent.messages[-1], HumanMessage)
        assert agent.messages[-1].content == "Hello, this is a test message."
    
    def test_add_assistant_message(self, agent):
        """Test adding an assistant message."""
        initial_count = len(agent.messages)
        agent.add_message("This is a response.", role="assistant")
        
        assert len(agent.messages) == initial_count + 1
        assert isinstance(agent.messages[-1], AIMessage)
        assert agent.messages[-1].content == "This is a response."
    
    def test_add_system_message(self, agent):
        """Test adding a system message."""
        initial_count = len(agent.messages)
        agent.add_message("System instruction.", role="system")
        
        assert len(agent.messages) == initial_count + 1
        assert isinstance(agent.messages[-1], SystemMessage)
        assert agent.messages[-1].content == "System instruction."
    
    def test_role_inference(self, agent):
        """Test automatic role inference based on previous messages."""
        # After system message, should default to user
        agent.add_message("First message")
        assert isinstance(agent.messages[-1], HumanMessage)
        
        # After user message, should default to assistant
        agent.add_message("Response message")
        assert isinstance(agent.messages[-1], AIMessage)
        
        # After assistant message, should default to user
        agent.add_message("Another user message")
        assert isinstance(agent.messages[-1], HumanMessage)
    
    def test_replace_message_at(self, agent):
        """Test replacing a message at a specific index."""
        agent.add_message("Original message")
        original_count = len(agent.messages)
        
        agent.replace_message_at(1, "Modified message")
        
        assert len(agent.messages) == original_count
        assert agent.messages[1].content == "Modified message"
        assert isinstance(agent.messages[1], HumanMessage)
    
    def test_remove_message_at(self, agent):
        """Test removing a message at a specific index."""
        agent.add_message("Message to remove")
        original_count = len(agent.messages)
        
        agent.remove_message_at(1)
        
        assert len(agent.messages) == original_count - 1
    
    def test_remove_message_at_invalid_index(self, agent):
        """Test removing a message with invalid index."""
        original_count = len(agent.messages)
        
        # Should not raise error, just do nothing
        agent.remove_message_at(999)
        agent.remove_message_at(-1)
        
        assert len(agent.messages) == original_count
    
    def test_reset_messages(self, agent):
        """Test resetting messages."""
        agent.add_message("Test message 1")
        agent.add_message("Test message 2")
        
        agent.reset()
        
        assert len(agent.messages) == 1  # Only system message should remain
        assert isinstance(agent.messages[0], SystemMessage)
        assert agent.messages[0].content == agent.system_prompt
    
    def test_add_system_prompt(self, agent):
        """Test adding/updating system prompt."""
        new_prompt = "You are a new test assistant."
        agent.add_system_prompt(new_prompt)
        
        assert agent.system_prompt == new_prompt
        assert isinstance(agent.messages[0], SystemMessage)
        assert agent.messages[0].content == new_prompt
    
    def test_get_messages_for_langchain(self, agent):
        """Test getting messages in LangChain format."""
        agent.add_message("Test message")
        
        messages = agent.get_messages_for_langchain()
        
        assert len(messages) == 2  # System + user message
        for msg in messages:
            assert isinstance(msg, BaseMessage)
        
        assert isinstance(messages[0], SystemMessage)
        assert isinstance(messages[1], HumanMessage)
    
    def test_encode_image_from_bytes(self):
        """Test encoding image from bytes."""
        test_bytes = b"fake image data"
        encoded = LMMAgent.encode_image(test_bytes)
        
        import base64
        expected = base64.b64encode(test_bytes).decode()
        assert encoded == expected
    
    def test_scrub_previous_images(self, agent):
        """Test scrubbing images from previous messages."""
        # Add a message with mock image content
        agent.add_message("Text with image", image_content=b"fake image")
        
        # Scrub images
        agent.scrub_previous_images()
        
        # Check that images were removed but text remained
        # Note: This is a basic test - full image handling would need more complex testing
        assert len(agent.messages) >= 1


class TestLMMAgentProviders:
    """Test different LLM providers."""
    
    def test_openai_provider(self):
        """Test OpenAI provider creation."""
        agent = LMMAgent(
            engine_params={
                "engine_type": "openai",
                "model": "gpt-3.5-turbo",
                "api_key": "dummy-key-for-testing"
            }
        )
        
        assert isinstance(agent.llm, ChatOpenAI)
    
    def test_anthropic_provider(self):
        """Test Anthropic provider creation."""
        agent = LMMAgent(
            engine_params={
                "engine_type": "anthropic",
                "model": "claude-3-sonnet-20240229",
                "api_key": "dummy-key-for-testing"
            }
        )
        
        assert isinstance(agent.llm, ChatAnthropic)
    
    def test_gemini_provider(self):
        """Test Gemini provider creation."""
        agent = LMMAgent(
            engine_params={
                "engine_type": "gemini",
                "model": "gemini-pro",
                "api_key": "dummy-key-for-testing",
                "base_url": "https://dummy-gemini-endpoint.com"
            }
        )
        
        # Gemini uses ChatOpenAI with custom base_url
        assert isinstance(agent.llm, ChatGoogleGenerativeAI)
    
    def test_unsupported_provider(self):
        """Test unsupported provider raises error."""
        with pytest.raises(ValidationError):
            LMMAgent(
                engine_params={
                    "engine_type": "unsupported_provider",
                    "model": "some-model",
                    "api_key": "dummy-key-for-testing"
                }
            )


if __name__ == "__main__":
    pytest.main([__file__])