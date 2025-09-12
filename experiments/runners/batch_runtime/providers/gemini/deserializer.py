from typing import Any, Dict, List, Optional

from agent.src.core.tools import AddToCartInput, ValidTools
from experiments.config import ExperimentId
from experiments.runners.batch_runtime.typedefs import (
    BatchResult,
    ExperimentFailureModes,
    ExperimentResult,
    ProviderBatchResult,
)

from .._base.deserializer import BaseBatchProviderDeserializer


class GeminiBatchProviderDeserializer(BaseBatchProviderDeserializer):
    """Functor for converting Gemini-specific batch results into a common batch result"""

    def deserialize(self, data: ProviderBatchResult) -> BatchResult:
        if "results" not in data:
            raise ValueError("No results found in the batch response")

        experiment_results: List[ExperimentResult] = []

        for result_data in data["results"]:
            experiment_result = self._process_single_result(result_data)
            experiment_results.append(experiment_result)

        return BatchResult(data=experiment_results)

    def _process_single_result(self, result_data: Dict[str, Any]) -> ExperimentResult:
        """Process a single Gemini batch result into an ExperimentResult."""
        # Extract custom_id
        custom_id = result_data.get("custom_id")
        if not custom_id:
            raise ValueError("No custom_id found in result")

        experiment_id = ExperimentId(custom_id)

        # Check for errors
        if "error" in result_data:
            error_msg = result_data["error"].get("message", "Unknown error")
            return ExperimentResult(
                experiment_id=experiment_id,
                response_content=f"API Error: {error_msg}",
                tool_call=None,
                failure_reason=ExperimentFailureModes.API_ERROR,
            )

        # Extract response content
        response = result_data.get("response", result_data)

        # Check if response has valid candidates
        candidates = response.get("candidates", [])
        if not candidates:
            return ExperimentResult(
                experiment_id=experiment_id,
                response_content="No candidates in response",
                tool_call=None,
                failure_reason=ExperimentFailureModes.API_ERROR,
            )

        # Extract content from first candidate
        candidate = candidates[0]
        content = candidate.get("content", {})
        parts = content.get("parts", [])

        # Extract text content and tool calls
        text_content = self._extract_text_content(parts)
        tool_call = self._extract_tool_call(parts)

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
    def _extract_text_content(parts: List[Dict[str, Any]]) -> str:
        """Extract text content from Gemini response parts."""
        text_parts = []
        for part in parts:
            if "text" in part:
                text_parts.append(part["text"])

        return " ".join(text_parts) if text_parts else ""

    @staticmethod
    def _extract_tool_call(parts: List[Dict[str, Any]]) -> Optional[AddToCartInput]:
        """Extract tool call from Gemini response parts."""
        for part in parts:
            if "functionCall" in part:
                func_call = part["functionCall"]
                func_name = func_call.get("name")

                if func_name == ValidTools.ADD_TO_CART:
                    args = func_call.get("args", {})
                    try:
                        # Convert Gemini args format to AddToCartInput
                        # Gemini may use different field names, so we need to map them
                        tool_input = AddToCartInput(
                            product_title=args.get("product_title", ""),
                            price=float(args.get("price", 0)),
                            rating=float(args.get("rating", 0)),
                            number_of_reviews=int(args.get("number_of_reviews", 0)),
                        )
                        return tool_input
                    except Exception:
                        # If parsing fails, return None (will be marked as NO_TOOL_CALL)
                        return None

        return None
