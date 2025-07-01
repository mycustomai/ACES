"""
Mock implementation of LMMAgent for testing.

This module provides a MockLMMAgent that mimics the behavior of the real LMMAgent
without making actual API calls, allowing for fast and deterministic testing.
"""

from typing import Any, Dict, Optional, List, Union, Literal
from unittest.mock import Mock
import json

from langchain_core.messages import (
    BaseMessage,
    SystemMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool

from agent.src.typedefs import EngineParams


class MockLMMAgent:
    """
    Mock implementation of LMMAgent for testing.
    
    This mock agent can be configured to return specific responses and tool calls
    without making actual API calls to language model providers.
    """
    
    def __init__(
        self,
        engine_params: Optional[Union[Dict[str, Any], EngineParams]] = None,
        system_prompt: Optional[str] = None,
        llm: Optional[Any] = None,
        tools: Optional[List[Union[BaseTool, Dict[str, Any]]]] = None,
        responses: Optional[List[Dict[str, Any]]] = None,
        default_tool_call: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the mock LMM Agent.
        
        Args:
            engine_params: Engine configuration parameters (ignored in mock)
            system_prompt: System prompt for the agent
            llm: Pre-configured LLM instance (ignored in mock)
            tools: List of tools available to the agent
            responses: List of pre-configured responses to return
            default_tool_call: Default tool call to use if no specific responses configured
        """
        # Store configuration
        self.engine_params = engine_params
        self.system_prompt = system_prompt or "You are a helpful assistant."
        self.tools = tools or []
        
        # Mock LLM
        self.llm = Mock()
        self._llm_with_tools = Mock()
        
        # Message history
        self.messages: List[BaseMessage] = []
        self.add_system_prompt(self.system_prompt)
        
        # Configure mock responses
        self.responses = responses or []
        self.response_index = 0
        
        # Default behavior for shopping agent - add mousepad to cart
        self.default_tool_call = default_tool_call or {
            "name": "add_to_cart",
            "args": {
                "product_name": "Gaming Mouse Pad Large",
                "product_price": 15.99,
                "justification": "This mouse pad meets the requirements for a gaming setup."
            },
            "id": "call_test_123"
        }
    
    def add_system_prompt(self, system_prompt: str) -> None:
        """Add or update the system prompt."""
        self.system_prompt = system_prompt
        
        # Remove existing system message if present
        self.messages = [msg for msg in self.messages if not isinstance(msg, SystemMessage)]
        
        # Add new system message at the beginning
        self.messages.insert(0, SystemMessage(content=system_prompt))
    
    def add_message(
        self,
        text_content: str,
        image_content: Optional[Union[str, bytes, List[Union[str, bytes]]]] = None,
        image_url: Optional[str] = None,
        role: Optional[Literal["user", "assistant", "system"]] = None,
        image_detail: str = "high",
        put_text_last: bool = False,
    ) -> None:
        """Add a new message to the list of messages."""
        
        # Infer role from previous message if not specified
        if role is None:
            if not self.messages or isinstance(self.messages[-1], SystemMessage):
                role = "user"
            elif isinstance(self.messages[-1], HumanMessage):
                role = "assistant"
            elif isinstance(self.messages[-1], AIMessage):
                role = "user"
            else:
                role = "user"
        
        # Create message content (simplified for mock)
        message_content = text_content
        if image_content or image_url:
            # In a real implementation, we'd handle image content properly
            # For mock, we just note that an image was included
            message_content = f"{text_content} [IMAGE_INCLUDED]"
        
        # Create appropriate message type
        if role == "user":
            message = HumanMessage(content=message_content)
        elif role == "assistant":
            message = AIMessage(content=message_content)
        elif role == "system":
            message = SystemMessage(content=message_content)
        else:
            raise ValueError(f"Unsupported role: {role}")
        
        self.messages.append(message)
    
    def add_tool_response(
        self, 
        tool_use_id: str, 
        response_content: str, 
        image_content: Optional[Union[str, bytes, List[Union[str, bytes]]]] = None
    ) -> None:
        """Add a tool response message."""
        content = response_content
        if image_content:
            content = f"{response_content} [IMAGE_INCLUDED]"
        
        message = ToolMessage(
            content=content,
            tool_call_id=tool_use_id
        )
        self.messages.append(message)
    
    def reset(self) -> None:
        """Reset messages to only contain the system prompt."""
        self.messages = []
        if self.system_prompt:
            self.messages.append(SystemMessage(content=self.system_prompt))
        self.response_index = 0
    
    def get_response(self) -> AIMessage:
        """Generate the next response based on previous messages."""
        return self._generate_response()
    
    async def aget_response(self) -> AIMessage:
        """Generate the next response based on previous messages asynchronously."""
        return self._generate_response()
    
    def _generate_response(self) -> AIMessage:
        """Internal method to generate a mock response."""
        # Use pre-configured response if available
        if self.response_index < len(self.responses):
            response_config = self.responses[self.response_index]
            self.response_index += 1
        else:
            # Use default response
            response_config = {
                "content": "I'll help you find a suitable product and add it to your cart.",
                "tool_calls": [self.default_tool_call]
            }
        
        # Create AI message with tool calls
        ai_message = AIMessage(
            content=response_config.get("content", ""),
            tool_calls=response_config.get("tool_calls", [])
        )
        
        # Add to message history
        self.messages.append(ai_message)
        
        return ai_message
    
    def get_messages_for_langchain(self) -> List[BaseMessage]:
        """Get messages in LangChain format."""
        return self.messages.copy()
    
    def raw_message_requests(self) -> list[dict]:
        """Get raw message requests for batch processing."""
        # Convert messages to a simple dict format for mock
        raw_messages = []
        for msg in self.messages:
            if isinstance(msg, SystemMessage):
                raw_messages.append({"role": "system", "content": msg.content})
            elif isinstance(msg, HumanMessage):
                raw_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                message_dict = {"role": "assistant", "content": msg.content}
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    message_dict["tool_calls"] = msg.tool_calls
                raw_messages.append(message_dict)
            elif isinstance(msg, ToolMessage):
                raw_messages.append({
                    "role": "tool",
                    "content": msg.content,
                    "tool_call_id": msg.tool_call_id
                })
        return raw_messages
    
    def configure_responses(self, responses: List[Dict[str, Any]]) -> None:
        """Configure a list of responses for the mock to return."""
        self.responses = responses
        self.response_index = 0
    
    def configure_shopping_response(
        self, 
        product_name: str = "Gaming Mouse Pad Large", 
        product_price: float = 15.99,
        justification: str = "This product meets the requirements."
    ) -> None:
        """Configure a typical shopping response."""
        tool_call = {
            "name": "add_to_cart",
            "args": {
                "product_name": product_name,
                "product_price": product_price,
                "justification": justification
            },
            "id": "call_test_123"
        }
        
        response = {
            "content": f"I found a suitable {product_name} for ${product_price}. {justification}",
            "tool_calls": [tool_call]
        }
        
        self.configure_responses([response])
    
    @staticmethod
    def encode_image(image_content: Union[str, bytes]) -> str:
        """Mock image encoding - returns a placeholder."""
        return "mock_base64_encoded_image_data"
    
    # Additional mock methods to match the real LMMAgent interface
    def remove_message_at(self, index: int) -> None:
        """Remove a message at a given index."""
        if 0 <= index < len(self.messages):
            self.messages.pop(index)
    
    def replace_message_at(
        self, 
        index: int, 
        text_content: str, 
        image_content: Optional[Union[str, bytes, List[Union[str, bytes]]]] = None, 
        image_detail: str = "high"
    ) -> None:
        """Replace a message at a given index."""
        if 0 <= index < len(self.messages):
            old_message = self.messages[index]
            
            # Create simplified content for mock
            new_content = text_content
            if image_content:
                new_content = f"{text_content} [IMAGE_INCLUDED]"
            
            if isinstance(old_message, HumanMessage):
                self.messages[index] = HumanMessage(content=new_content)
            elif isinstance(old_message, AIMessage):
                self.messages[index] = AIMessage(content=new_content)
            elif isinstance(old_message, SystemMessage):
                self.messages[index] = SystemMessage(content=new_content)
    
    def scrub_previous_images(self) -> None:
        """Remove any image content from all previous messages (mock implementation)."""
        # In mock, we just remove [IMAGE_INCLUDED] markers
        for i, message in enumerate(self.messages):
            if hasattr(message, 'content') and isinstance(message.content, str):
                cleaned_content = message.content.replace(" [IMAGE_INCLUDED]", "")
                if isinstance(message, HumanMessage):
                    self.messages[i] = HumanMessage(content=cleaned_content)
                elif isinstance(message, AIMessage):
                    self.messages[i] = AIMessage(content=cleaned_content)
                elif isinstance(message, ToolMessage):
                    self.messages[i] = ToolMessage(
                        content=cleaned_content,
                        tool_call_id=message.tool_call_id
                    )