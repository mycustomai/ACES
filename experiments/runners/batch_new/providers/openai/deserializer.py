from typing import List

from openai.types.chat import ChatCompletionMessageToolCall

from agent.src.core.tools import AddToCartInput, ValidTools
from experiments.config import ExperimentId
from experiments.runners.batch_new.typedefs import (BatchResult,
                                                    ExperimentFailureModes,
                                                    ExperimentResult,
                                                    ProviderBatchResult)

from .._base.deserializer import BaseBatchProviderDeserializer
from .models import OpenAIBatchResultLine, OpenAIBatchResults


class OpenAIBatchProviderDeserializer(BaseBatchProviderDeserializer):
    """Functor for converting provider-specific batch results into a common batch result"""

    def deserialize(self, data: ProviderBatchResult) -> BatchResult:
        batch_results = OpenAIBatchResults.model_validate(data)
        if not batch_results.results:
            raise ValueError("No batch results found in the response")

        experiment_results: list[ExperimentResult] = [
            self._process_single_result(result) for result in batch_results.results
        ]

        return BatchResult(data=experiment_results)

    def _process_single_result(self, result: OpenAIBatchResultLine) -> ExperimentResult:
        """Process a single OpenAI batch result into an ExperimentResult"""
        failure_reason = None
        tool_call_data = None
        if result.has_error():
            response_content = f"API Error message: {result.response.error.message}"
            failure_reason = ExperimentFailureModes.API_ERROR
        else:
            response_content = result.get_content()
            tool_calls = result.get_tool_calls()
            if tool_calls:
                tool_call_data = self._extract_tool_call(tool_calls)
            else:
                failure_reason = ExperimentFailureModes.NO_TOOL_CALL

        return ExperimentResult(
            experiment_id=ExperimentId(result.custom_id),
            response_content=response_content,
            tool_call=tool_call_data,
            failure_reason=failure_reason,
        )

    @staticmethod
    def _extract_tool_call(
        tool_calls: List[ChatCompletionMessageToolCall],
    ) -> AddToCartInput:
        """Extract and convert tool call to AddToCartInput"""
        for tool_call in tool_calls:
            # noinspection PyUnreachableCode
            match tool_call.function.name:
                case ValidTools.ADD_TO_CART:
                    return AddToCartInput.model_validate_json(
                        tool_call.function.arguments
                    )
                case _:
                    raise ValueError(f"Unknown tool call: {tool_call.function.name}")

        raise ValueError("No add_to_cart tool call found in the response")
