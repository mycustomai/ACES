import base64
import backoff
from typing import Any, Dict, Optional, List, Union, Literal, cast
from warnings import warn

# noinspection PyProtectedMember
from langchain_anthropic.chat_models import _format_messages
from langchain_anthropic import ChatAnthropic
from langchain_aws import ChatBedrock
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    BaseMessage,
    SystemMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
    convert_to_openai_messages
)
from langchain_core.tools import BaseTool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import ChatHuggingFace
from langchain_openai import ChatOpenAI

from agent.src.config import get_config
from agent.src.typedefs import EngineParams, EngineType
from common.messages import RawMessageExchange


class LMMAgent:
    """LangChain-based Language Model Agent with strong typing and multi-provider support."""
    
    llm: BaseChatModel
    _llm_with_tools: BaseChatModel
    messages: List[SystemMessage | HumanMessage | AIMessage | ToolMessage]
    system_prompt: Optional[str]
    tools: List[BaseTool]

    def __init__(
        self,
        engine_params: Optional[Union[Dict[str, Any], EngineParams]] = None,
        system_prompt: Optional[str] = None,
        llm: Optional[BaseChatModel] = None,
        tools: Optional[List[Union[BaseTool, Dict[str, Any]]]] = None,
    ):
        """Initialize the LangChain-based LMM Agent.
        
        Args:
            engine_params: Engine configuration parameters (dict or EngineParams)
            system_prompt: System prompt for the agent
            llm: Pre-configured LangChain LLM instance
            tools: List of tools (BaseTool instances or dict definitions)
        """
        if llm is None:
            if engine_params is not None:
                self.llm = LMMAgent._create_llm_from_params(engine_params)
            else:
                raise ValueError("Either llm or engine_params must be provided")
        else:
            self.llm = llm

        self.messages = []
        self.tools: List[BaseTool] = LMMAgent._process_tools(tools or [])

        self._llm_with_tools = cast(BaseChatModel, self.llm.bind_tools(self.tools))

        # Set system prompt
        self.system_prompt = system_prompt or "You are a helpful assistant."
        self.add_system_prompt(self.system_prompt)

    @staticmethod
    def _create_llm_from_params(engine_params: Union[Dict[str, Any], EngineParams]) -> BaseChatModel:
        """Create LangChain LLM from engine parameters."""
        # Coerce dict to EngineParams for validation
        if isinstance(engine_params, dict):
            params = EngineParams.from_dict(engine_params)
        else:
            params = engine_params
            
        engine_type = params.engine_type
        model = params.model
        api_key = params.api_key
        temperature = params.temperature
        max_tokens = params.max_new_tokens
        
        # Get configuration for API key fallbacks
        config = get_config()

        if engine_type == EngineType.BEDROCK:
            params_dict = params.to_dict()
            # noinspection Pydantic
            return ChatBedrock(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                **params_dict,
            )
        elif engine_type == EngineType.OPENAI:
            # Extract OpenAI-specific parameters
            openai_kwargs = {}
            if max_tokens:
                openai_kwargs["max_tokens"] = max_tokens
            
            # Add provider-specific fields if present in the original data
            params_dict = params.to_dict()
            for field in ["top_p", "frequency_penalty", "presence_penalty", "stop"]:
                if field in params_dict and params_dict[field] is not None:
                    openai_kwargs[field] = params_dict[field]
            
            final_api_key = api_key or config.openai_api_key
            if not final_api_key:
                raise ValueError("No OpenAI API key provided. Set OPENAI_API_KEY environment variable or provide api_key parameter.")
            
            return ChatOpenAI(
                model=model,
                api_key=final_api_key,
                temperature=temperature,
                **openai_kwargs
            )
        elif engine_type == EngineType.ANTHROPIC:
            # ChatAnthropic uses max_tokens_to_sample instead of max_tokens
            anthropic_kwargs = {}
            if max_tokens:
                anthropic_kwargs["max_tokens_to_sample"] = max_tokens
            
            # Add Anthropic-specific fields if present in the original data
            params_dict = params.to_dict()
            # Note: 'thinking' is handled specially in the engine, not passed to ChatAnthropic
            for field in ["top_p", "top_k"]:
                if field in params_dict and params_dict[field] is not None:
                    anthropic_kwargs[field] = params_dict[field]
            
            final_api_key = api_key or config.anthropic_api_key
            if not final_api_key:
                raise ValueError("No Anthropic API key provided. Set ANTHROPIC_API_KEY environment variable or provide api_key parameter.")
            
            return ChatAnthropic(
                model_name=model,
                api_key=final_api_key,
                temperature=temperature,
                **anthropic_kwargs
            )
        elif engine_type == EngineType.GEMINI:
            # Use the dedicated Google Gemini LangChain integration
            gemini_kwargs = {}
            if max_tokens:
                gemini_kwargs["max_output_tokens"] = max_tokens
            
            # Add Gemini-specific fields if present in the original data
            params_dict = params.to_dict()
            for field in ["safety_settings", "top_p", "top_k"]:
                if field in params_dict and params_dict[field] is not None:
                    gemini_kwargs[field] = params_dict[field]
            
            final_api_key = api_key or config.google_api_key
            if not final_api_key:
                raise ValueError("No Google API key provided. Set GOOGLE_API_KEY environment variable or provide api_key parameter.")
            
            return ChatGoogleGenerativeAI(
                model=model,
                google_api_key=final_api_key,
                temperature=temperature,
                # TODO: ensure that model-specific config YAML parses this correctly
                #thinking_budget=0,
                max_retries=1,
                timeout=10,
                **gemini_kwargs,
            )
        else:
            raise ValueError(f"Unsupported engine_type: {engine_type}")

    # TODO: remove this functionality
    @staticmethod
    def _process_tools(tools: List[Union[BaseTool, Dict[str, Any]]]) -> List[BaseTool]:
        """Process and convert tools to LangChain BaseTool instances."""
        processed_tools = []
        for tool in tools:
            if isinstance(tool, BaseTool):
                processed_tools.append(tool)
            elif isinstance(tool, dict):
                # For backward compatibility, we'll skip dict tools for now
                # In a real implementation, you'd convert these to proper BaseTool instances
                continue
        return processed_tools
    
    @staticmethod
    def encode_image(image_content: Union[str, bytes]) -> str:
        """Encode image content to base64 string."""
        if isinstance(image_content, str):
            with open(image_content, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode()
        else:
            return base64.b64encode(image_content).decode()

    def reset(self) -> None:
        """Reset messages to only contain the system prompt."""
        self.messages = []
        if self.system_prompt:
            self.messages.append(SystemMessage(content=self.system_prompt))

    def add_system_prompt(self, system_prompt: str) -> None:
        """Add or update the system prompt."""
        self.system_prompt = system_prompt
        
        # Remove existing system message if present
        self.messages = [msg for msg in self.messages if not isinstance(msg, SystemMessage)]
        
        # Add new system message at the beginning
        self.messages.insert(0, SystemMessage(content=system_prompt))

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
            
            # Create content with text
            if isinstance(old_message, HumanMessage):
                new_content = text_content
                if image_content:
                    # For images, we'll use the content format that LangChain supports
                    content_parts = [{"type": "text", "text": text_content}]
                    
                    if isinstance(image_content, list):
                        for image in image_content:
                            base64_image = self.encode_image(image)
                            content_parts.append({
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}",
                                    "detail": image_detail,
                                },
                            })
                    else:
                        base64_image = self.encode_image(image_content)
                        content_parts.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}",
                                "detail": image_detail,
                            },
                        })
                    new_content = content_parts
                
                self.messages[index] = HumanMessage(content=new_content)
            elif isinstance(old_message, AIMessage):
                self.messages[index] = AIMessage(content=text_content)
            elif isinstance(old_message, SystemMessage):
                self.messages[index] = SystemMessage(content=text_content)

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
        
        # Create message content
        if image_url:
            content_parts = [{"type": "text", "text": text_content}]
            
            # Provider-specific image URL formatting
            if isinstance(self.llm, ChatOpenAI):
                # OpenAI format
                content_parts.append({
                    "type": "image_url",
                    "image_url": {
                        "url": image_url,
                        "detail": image_detail,
                    },
                })
            elif isinstance(self.llm, ChatAnthropic):
                # Anthropic format
                content_parts.append({
                    "type": "image",
                    "source": {
                        "type": "url",
                        "url": image_url,
                    },
                })
            elif isinstance(self.llm, ChatGoogleGenerativeAI):
                # Google Gemini format - uses image_url similar to OpenAI
                content_parts.append({
                    "type": "image_url",
                    "image_url": {
                        "url": image_url,
                    },
                })
            elif isinstance(self.llm, ChatHuggingFace):
                # HuggingFace format - varies by model, using OpenAI-like format as default
                content_parts.append({
                    "type": "image_url",
                    "image_url": {
                        "url": image_url,
                        "detail": image_detail,
                    },
                })
            else:
                # Fallback to OpenAI format for unknown providers
                content_parts.append({
                    "type": "image_url",
                    "image_url": {
                        "url": image_url,
                        "detail": image_detail,
                    },
                })
            
            message_content = content_parts
        elif image_content:
            content_parts = [{"type": "text", "text": text_content}]
            
            # Handle multiple images
            if isinstance(image_content, list):
                for image in image_content:
                    base64_image = self.encode_image(image)
                    content_parts.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}",
                            "detail": image_detail,
                        },
                    })
            else:
                # Single image
                base64_image = self.encode_image(image_content)
                content_parts.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}",
                        "detail": image_detail,
                    },
                })
            
            # Rotate text to be last if desired
            if put_text_last:
                text_part = content_parts.pop(0)
                content_parts.append(text_part)
            
            message_content = content_parts
        else:
            message_content = text_content
        
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

    def scrub_previous_images(self) -> None:
        """
        Remove any image content from all previous messages.
        This method iterates over self.messages and removes image content.
        """
        def scrub_content(content):
            if isinstance(content, list):
                new_content = []
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") not in ["image", "image_url"]:
                            # Keep non-image content
                            if item.get("type") == "tool_result" and "content" in item:
                                item["content"] = scrub_content(item["content"])
                            new_content.append(item)
                    else:
                        new_content.append(item)
                return new_content
            return content
        
        for i, message in enumerate(self.messages):
            if hasattr(message, 'content'):
                scrubbed_content = scrub_content(message.content)
                if isinstance(message, HumanMessage):
                    self.messages[i] = HumanMessage(content=scrubbed_content)
                elif isinstance(message, AIMessage):
                    self.messages[i] = AIMessage(content=scrubbed_content)
                elif isinstance(message, ToolMessage):
                    self.messages[i] = ToolMessage(
                        content=scrubbed_content,
                        tool_call_id=message.tool_call_id
                    )

    def add_tool_response(
        self, 
        tool_use_id: str, 
        response_content: str, 
        image_content: Optional[Union[str, bytes, List[Union[str, bytes]]]] = None
    ) -> None:
        """Add a tool response message.
        
        If image_content is provided, this method will scrub old messages of any image data.
        """
        # If there's image content, scrub all previous messages to remove images first
        if image_content:
            self.scrub_previous_images()
        
        # For LangChain, we'll use ToolMessage
        content = response_content
        
        if image_content:
            # Create content with text and images
            content_parts = [{"type": "text", "text": response_content}]
            
            if isinstance(image_content, list):
                for image in image_content:
                    base64_image = self.encode_image(image)
                    content_parts.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": base64_image,
                        },
                    })
            else:
                base64_image = self.encode_image(image_content)
                content_parts.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": base64_image,
                    },
                })
            content = content_parts
        
        message = ToolMessage(
            content=content,
            tool_call_id=tool_use_id
        )
        self.messages.append(message)

    def _handle_response(self, response: BaseMessage) -> AIMessage:
        """Handle the model response - validate, store in history, and return.
        
        Args:
            response: The raw response from the LLM
            
        Returns:
            The validated AI message
            
        Raises:
            ValueError: If the response is not a valid AIMessage or lacks tool calls
        """
        if not isinstance(response, AIMessage):
            raise ValueError("Invalid response type. Expected AIMessage, got {}".format(type(response)))
    
        # Add response to message history
        self.messages.append(response)
    
        # Validate response
        if len(response.tool_calls) > 1:
            warn("Model returned more than 1 tool.")
    
        return response
    
    @backoff.on_exception(backoff.expo, Exception, max_tries=5)
    def get_response(
        self,
    ) -> AIMessage:
        """Generate the next response based on previous messages."""
        response: BaseMessage = self._llm_with_tools.invoke(self.messages)
        return self._handle_response(response)
    
    @backoff.on_exception(backoff.fibo, Exception, max_time=60)
    async def aget_response(
        self,
    ) -> AIMessage:
        """Generate the next response based on previous messages asynchronously."""
        response: BaseMessage = await self._llm_with_tools.ainvoke(self.messages)
        return self._handle_response(response)
    
    def get_messages_for_langchain(self) -> List[BaseMessage]:
        """Get messages in LangChain format."""
        return self.messages.copy()

    def raw_message_requests(self) -> RawMessageExchange:
        """ Get raw message requests for batch processing - specific to each engine """
        if isinstance(self.llm, ChatOpenAI) or isinstance(self.llm, ChatGoogleGenerativeAI):
            return RawMessageExchange(convert_to_openai_messages(self.messages))
        elif isinstance(self.llm, ChatAnthropic):
            system_msg, formatted_messages = _format_messages(self.messages)
            
            if system_msg:
                formatted_messages = [{"role": "system", "content": system_msg}] + formatted_messages

            return RawMessageExchange(formatted_messages)
        elif isinstance(self.llm, ChatHuggingFace):
            raise NotImplementedError()
        else:
            raise ValueError("Unsupported engine type")
