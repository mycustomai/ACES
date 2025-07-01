import asyncio
from typing import Dict, Any, Optional

import backoff
from langchain_core.messages import AIMessage

from agent.src.core.mllm import LMMAgent
from agent.src.core.tools import AddToCartTool, AddToCartInput
from agent.src.environment import BaseShoppingEnvironment
from agent.src.logger import ExperimentLogger
from agent.src.typedefs import EngineParams
from common.messages import RawMessageExchange


class NoToolCallException(ValueError):
    pass


class SimulatedShopper:
    """
    A simulated shopper that uses an LLM agent to interact with a shopping website.
    """
    initial_message: str
    agent: LMMAgent
    environment: BaseShoppingEnvironment
    logger: Optional[ExperimentLogger]

    def __init__(
            self,
            initial_message: str,
            engine_params: EngineParams,
            environment: BaseShoppingEnvironment,
            logger: Optional[ExperimentLogger] = None,
    ):
        """
        Initialize the SimulatedShopper with an initial message and LLM engine parameters.
        
        Args:
            initial_message (str): The initial message to send to the agent.
            engine_params (EngineParams): Parameters for the LLM engine.
            environment (BaseShoppingEnvironment): The shopping environment to interact with.
            logger (Optional[ExperimentLogger]): An optional logger for recording interactions.
        """
        self.initial_message = initial_message
        
        tools = [
            AddToCartTool()
        ]
        
        self.agent = LMMAgent(
            engine_params=engine_params,
            system_prompt="""You are a helpful shopping assistant and are to purchase a single product.""",
            tools=tools
        )

        self.environment = environment
        self.logger = logger

    def _execute_tool_call(self, response: AIMessage) -> bool:
        if not response.tool_calls:
            raise NoToolCallException("AIMessage does not contain tool calls")

        tool_call = response.tool_calls[0]
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]

        match tool_name:
            case "add_to_cart":
                tool_args_obj: AddToCartInput = AddToCartInput.model_validate(tool_args)
                if self.logger:
                    self.logger.record_cart_item(tool_args_obj)
                return True

        return False

    def _prepare_agent(self):
        if not self.environment:
            raise RuntimeError("Environment not set")

        # first message and log initial data
        if hasattr(self.environment, 'remote') and self.environment.remote:
            screenshot_url = self.environment.capture_screenshot()
            self.agent.add_message(
                text_content=self.initial_message,
                image_url=screenshot_url,
                role="user"
            )


            if self.logger:
                self.logger.record_initial_interaction(screenshot_url=screenshot_url, text=self.initial_message)
        else:
            screenshot_bytes = self.environment.capture_screenshot()
            self.agent.add_message(
                text_content=self.initial_message,
                image_content=screenshot_bytes,
                role="user"
            )

            if self.logger:
                self.logger.record_initial_interaction(screenshot=screenshot_bytes, text=self.initial_message)

    @backoff.on_exception(backoff.constant, ValueError, max_tries=5, interval=1)
    async def _get_response(self) -> AIMessage:
        """ Add retry-loop in the event that a model does not return a tool call """
        return await self.agent.get_response()

    def run(self):
        self._prepare_agent()

        while True:
            response = self.agent.get_response()
            self.logger.record_agent_interaction(response)

            end_loop = self._execute_tool_call(response)
            if end_loop:
                break

        self.logger.create_screenshots_gif()

   #@backoff.on_exception(backoff.constant, ValueError, max_tries=5, interval=1)
    async def _aget_response(self) -> AIMessage:
        """ Add a retry-loop in the event that a model does not return a tool call """
        return await self.agent.aget_response()

    async def arun(self):
        self._prepare_agent()

        while True:
            response = await self._aget_response()
            if self.logger:
                self.logger.record_agent_interaction(response)

            end_loop = self._execute_tool_call(response)
            if end_loop:
                break
                
            # Yield control to allow other tasks to run
            await asyncio.sleep(0)

        if self.logger:
            self.logger.create_screenshots_gif()

    def get_batch_request(self) -> RawMessageExchange:
        self._prepare_agent()

        return self.agent.raw_message_requests()