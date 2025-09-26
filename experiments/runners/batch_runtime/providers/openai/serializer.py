from typing import Any

from agent.src.typedefs import EngineParams
from common.messages import RawMessageExchange
from experiments.runners.batch_runtime.typedefs import (BatchRequest,
                                                        ProviderBatchRequest,
                                                        SerializedBatchRequest)

from .._base.serializer import BaseBatchProviderSerializer


class OpenAIBatchProviderSerializer(BaseBatchProviderSerializer):
    """Functor for converting common batch types to provider specific types"""

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
    def _create_single_request(
        raw_messages: RawMessageExchange,
        engine_params: EngineParams,
        custom_id: str,
        tools: list[dict[str, Any]] = None,
    ) -> ProviderBatchRequest:
        """Serialization of a single experiment."""
        body: dict[str, Any] = {
            "model": engine_params.model,
            "messages": raw_messages,
        }

        if engine_params.temperature is not None:
            body["temperature"] = engine_params.temperature

        if engine_params.max_new_tokens is not None:
            body["max_tokens"] = engine_params.max_new_tokens

        batch_request: dict[str, Any] = {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": body,
        }

        if tools:
            batch_request["body"]["tools"] = tools
            batch_request["body"]["tool_choice"] = "auto"

        reasoning_effort = getattr(engine_params, "reasoning_effort", None)
        if reasoning_effort is not None:
            batch_request["body"]["reasoning_effort"] = reasoning_effort

        return ProviderBatchRequest(batch_request)
