from typing import List, Optional

from anthropic.types.beta import (BetaContentBlock, BetaTextBlock,
                                  BetaToolUseBlock)
from anthropic.types.beta.messages import (BetaMessageBatchCanceledResult,
                                           BetaMessageBatchErroredResult,
                                           BetaMessageBatchExpiredResult,
                                           BetaMessageBatchIndividualResponse,
                                           BetaMessageBatchResult,
                                           BetaMessageBatchSucceededResult)
from pydantic import ValidationError
from rich import print as _print

from agent.src.core.tools import AddToCartInput, ValidTools
from experiments.config import ExperimentId
from experiments.runners.batch_runtime.common.encoded_id_mixin import \
    EncodedExperimentIdMixin
from experiments.runners.batch_runtime.typedefs import (BatchResult,
                                                        ExperimentFailureModes,
                                                        ExperimentResult,
                                                        ProviderBatchResult)

from .._base.deserializer import BaseBatchProviderDeserializer


class AnthropicBatchProviderDeserializer(
    BaseBatchProviderDeserializer, EncodedExperimentIdMixin
):
    """Functor for converting provider-specific batch results into a common batch result"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def deserialize(self, batch_data: ProviderBatchResult) -> BatchResult:
        """Convert Anthropic batch results to common BatchResult format."""
        if "results" not in batch_data:
            raise ValueError("No results found in the batch response")

        experiment_results: List[ExperimentResult] = []

        for result_data in batch_data["results"]:
            parsed_result_data = BetaMessageBatchIndividualResponse.model_validate(
                result_data
            )
            experiment_result = self._process_single_result(parsed_result_data)
            experiment_results.append(experiment_result)

        return BatchResult(data=experiment_results)

    def _process_single_result(
        self, result_data: BetaMessageBatchIndividualResponse
    ) -> ExperimentResult:
        """Process a single Anthropic batch result into an ExperimentResult."""
        # TODO: test that custom_id is encoded

        # The custom_id is passed through as-is, allowing the ExperimentLoader
        # to handle decoding through its get_experiment_by_id method which supports
        # both encoded hash-based IDs and regular string IDs.
        experiment_id = ExperimentId(result_data.custom_id)

        result = result_data.result
        handlers = {
            BetaMessageBatchSucceededResult: self._process_success_result,
            BetaMessageBatchErroredResult: self._process_error_result,
            BetaMessageBatchCanceledResult: self._process_error_result,
            BetaMessageBatchExpiredResult: self._process_error_result,
        }
        for result_type, handler in handlers.items():
            if isinstance(result, result_type):
                return handler(experiment_id, result)

        raise ValueError(f"Unknown result type: {type(result)}")

    def _process_success_result(
        self, experiment_id: ExperimentId, result: BetaMessageBatchSucceededResult
    ) -> ExperimentResult:
        """Process a successful Anthropic result."""
        message = result.message
        content = message.content

        text_content = self._extract_text_content(content)
        tool_call = self._extract_tool_call(content)

        failure_reason = None
        if not tool_call:
            failure_reason = ExperimentFailureModes.NO_TOOL_CALL

        return ExperimentResult(
            experiment_id=experiment_id,
            response_content=text_content,
            tool_call=tool_call,
            failure_reason=failure_reason,
        )

    @staticmethod
    def _process_error_result(
        experiment_id: ExperimentId, result: BetaMessageBatchResult
    ) -> ExperimentResult:
        """Process an error result from Anthropic."""
        error_type = result.type
        response_content = f"API Error: Result is {error_type}"

        return ExperimentResult(
            experiment_id=experiment_id,
            response_content=response_content,
            tool_call=None,
            failure_reason=ExperimentFailureModes.API_ERROR,
        )

    @staticmethod
    def _extract_text_content(content: list[BetaContentBlock]) -> str:
        """Extract text content from the Anthropic response content list.

        Raises:
            ValueError: If no or multiple text blocks are found in the response.
        """
        text = None
        for item in content:
            if isinstance(item, BetaTextBlock):
                if text is not None:
                    raise ValueError("Multiple text blocks found in the response")
                text = item.text
        if text is None:
            raise ValueError("No text blocks found in the response")
        return text

    @staticmethod
    def _extract_tool_call(content: list[BetaContentBlock]) -> Optional[AddToCartInput]:
        """Extract tool call from Anthropic response content array."""
        tool_calls = []
        for item in content:
            if isinstance(item, BetaToolUseBlock):
                tool_calls.append(item)

        if len(tool_calls) > 1:
            raise ValueError("Multiple tool calls found in the response")

        if len(tool_calls) == 0:
            return None

        tool_block = tool_calls[0]
        tool_name = tool_block.name
        if tool_name == ValidTools.ADD_TO_CART:
            tool_input = tool_block.input
            try:
                return AddToCartInput.model_validate(tool_input)
            except ValidationError:
                _print("[red bold] Failed to parse tool input: {tool_input}")
        return None
