from typing import Any

from anthropic.types import ToolParam
from anthropic.types.message_create_params import \
    MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request
from anthropic.types.tool_param import InputSchema

from experiments.config import ExperimentId
from experiments.runners.batch_runtime.common.encoded_id_mixin import \
    EncodedExperimentIdMixin
from experiments.runners.batch_runtime.typedefs import (BatchRequest,
                                                        ProviderBatchRequest,
                                                        SerializedBatchRequest)

from .._base.serializer import BaseBatchProviderSerializer


class AnthropicBatchProviderSerializer(
    BaseBatchProviderSerializer, EncodedExperimentIdMixin
):
    """
    Functor for converting common batch types to provider-specific types.

    Uses Anthropic SDK's MessageCreateParamsNonStreaming and Request types
    for full type safety and API compliance.
    """

    DEFAULT_MAX_NEW_TOKENS = 8096
    """ Required by Anthropic Batch API. """

    def serialize(self, data: BatchRequest) -> list[SerializedBatchRequest]:
        serialized_requests = []
        tools_dict = data.tool_request_dict()

        for messages, experiment in zip(data.raw_messages, data.experiments):
            provider_request = self._create_single_request(
                raw_messages=messages,
                engine_params=data.engine_params,
                tools=tools_dict if tools_dict else None,
                experiment_id=experiment.experiment_id,
            )
            serialized_requests.append(
                SerializedBatchRequest(
                    experiment_id=experiment.experiment_id,
                    provider_request=provider_request,
                )
            )
        return serialized_requests

    def _create_single_request(
        self,
        raw_messages: list[dict[str, Any]],
        engine_params: Any,
        experiment_id: ExperimentId,
        tools: list[dict[str, Any]] = None,
    ) -> ProviderBatchRequest:
        """Serialization of a single experiment."""
        custom_id = self.encode_custom_id(experiment_id)

        system_content = None
        user_messages = []

        for msg in raw_messages:
            if msg.get("role") == "system":
                system_content = msg.get("content")
            else:
                user_messages.append(msg)

        params: MessageCreateParamsNonStreaming = {
            "model": engine_params.model,
            "max_tokens": engine_params.max_new_tokens or self.DEFAULT_MAX_NEW_TOKENS,
            "messages": user_messages,
        }

        # add optional parameters
        if engine_params.temperature:
            params["temperature"] = engine_params.temperature
        if system_content:
            params["system"] = system_content

        if tools:
            anthropic_tools = self._convert_tools_to_anthropic(tools)
            if anthropic_tools:
                params["tools"] = anthropic_tools
            else:
                raise ValueError(
                    "Could not convert tools to Anthropic format. Check tool schema format."
                )

        batch_request: Request = {"custom_id": custom_id, "params": params}

        return ProviderBatchRequest(dict(batch_request))

    @staticmethod
    def _convert_tools_to_anthropic(tools: list[dict[str, Any]]) -> list[ToolParam]:
        """Convert OpenAI tool schema format to Anthropic tool format."""
        anthropic_tools = []

        for tool in tools:
            if tool.get("type") != "function":
                continue

            function_def = tool.get("function", {})
            input_schema: InputSchema = function_def.get("parameters", {})

            if "type" not in input_schema:
                input_schema["type"] = "object"
            if "properties" not in input_schema:
                input_schema["properties"] = {}

            name = function_def.get("name")
            description = function_def.get("description")

            if not isinstance(name, str) or not isinstance(description, str):
                raise ValueError(
                    f"Invalid tool schema: name and description must be strings. Got: {name}, {description}"
                )

            anthropic_tool: ToolParam = {
                "name": name,
                "description": description,
                "input_schema": input_schema,
            }

            anthropic_tools.append(anthropic_tool)

        return anthropic_tools
