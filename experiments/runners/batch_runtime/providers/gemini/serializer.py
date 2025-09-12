from typing import Any, Optional

from google.genai import types

from agent.src.typedefs import GeminiParams
from common.messages import RawMessageExchange
from experiments.runners.batch_runtime.typedefs import (
    BatchRequest,
    ProviderBatchRequest,
    SerializedBatchRequest,
)

from .._base.serializer import BaseBatchProviderSerializer


def _model_dump(x, **kwargs):
    return x.model_dump(
        mode="json",
        by_alias=True,
        exclude_none=True,
        exclude_unset=True,
        exclude_defaults=True,
        **kwargs,
    )


class GeminiBatchProviderSerializer(BaseBatchProviderSerializer):
    """Functor for converting common batch types to Gemini-specific types"""

    engine_params: GeminiParams

    def serialize(self, data: BatchRequest) -> list[SerializedBatchRequest]:
        serialized_requests = []
        tools_dict = data.tool_request_dict()

        for messages, experiment in zip(data.raw_messages, data.experiments):
            provider_request = self._create_single_request(
                raw_messages=messages,
                engine_params=self.engine_params,
                tools=tools_dict if tools_dict else None,
                custom_id=experiment.experiment_id,
            )
            serialized_requests.append(
                SerializedBatchRequest(
                    experiment_id=experiment.experiment_id,
                    provider_request=provider_request,
                )
            )

        return serialized_requests

    @staticmethod
    def _make_part(raw_part: dict) -> types.Part:
        if raw_part["type"] == "text":
            return types.Part.from_text(text=raw_part["text"])
        if raw_part["type"] == "image_url":
            img_url = raw_part["image_url"]["url"]
            if img_url.startswith("https://storage.googleapis.com/"):
                uri = img_url.replace("https://storage.googleapis.com/", "gs://")
            elif img_url.startswith("gs://"):
                uri = img_url
            else:
                raise ValueError(f"Invalid image URL format: {img_url}")
            return types.Part.from_uri(file_uri=uri)
        raise ValueError(f"Unknown part type {raw_part['type']}")

    @classmethod
    def _create_single_request(
        cls,
        raw_messages: RawMessageExchange,
        engine_params: GeminiParams,
        custom_id: str,
        tools: list[dict[str, Any]] = None,
    ) -> ProviderBatchRequest:
        """Serialization of a single experiment to Gemini GenerateContentRequest format."""
        # Convert raw_messages to Gemini GenerateContentRequest format
        contents: list[types.Content] = []
        system_instruction: Optional[types.Content] = None

        for msg in raw_messages:  # type: dict[str, Any]
            # Skip adding system messages to contents
            if msg["role"] == "system":
                raw_system_instruction = (
                    msg["content"]
                    if isinstance(msg["content"], str)
                    else "\n".join(
                        p["text"] for p in msg["content"] if p["type"] == "text"
                    )
                )
                system_instruction = types.Content(
                    parts=[types.Part.from_text(text=raw_system_instruction)]
                )
                continue

            parts = None

            if isinstance(msg.get("content"), str):
                parts = [types.Part.from_text(text=msg["content"])]
            elif isinstance(msg.get("content"), list):
                parts = [cls._make_part(p) for p in msg["content"]]
            else:
                # ignore extra
                pass

            if parts:
                contents.append(
                    types.Content(
                        parts=parts,
                        role="model" if msg["role"] == "assistant" else msg["role"],
                    )
                )
            else:
                raise ValueError(
                    f"Unexpected message format. Parsed empty contents: {msg}"
                )

        gemini_tools = None
        if tools:
            gemini_tools = cls._convert_tools_to_gemini(tools)

        thinking_config = (
            types.GenerationConfigThinkingConfig(
                thinking_budget=engine_params.thinking_budget,
            )
            if engine_params.thinking_budget is not None
            else None
        )

        config = types.GenerationConfig(
            max_output_tokens=engine_params.max_new_tokens,
            temperature=engine_params.temperature,
            thinking_config=thinking_config,
        )

        request = {
            "contents": [_model_dump(p) for p in contents],
            "generationConfig": _model_dump(config),
        }

        if system_instruction:
            request["systemInstruction"] = _model_dump(system_instruction)
        if gemini_tools:
            request["tools"] = [_model_dump(tool) for tool in gemini_tools]

        wrapped_request = {
            "request": request,
            "custom_id": custom_id,
        }

        return ProviderBatchRequest(wrapped_request)

    @staticmethod
    def _convert_tools_to_gemini(tools: list[dict[str, Any]]) -> list[types.Tool]:
        """Convert OpenAI tool schema format to Gemini function calling format."""
        return [
            types.Tool(
                function_declarations=[
                    types.FunctionDeclaration(
                        name=tool["function"]["name"],
                        description=tool["function"]["description"],
                        parameters=tool["function"]["parameters"],
                    )
                ]
            )
            for tool in tools
        ]
