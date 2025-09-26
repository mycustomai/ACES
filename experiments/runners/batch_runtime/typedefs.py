from enum import StrEnum
from typing import Any, NewType, Optional

from langchain_core.tools import BaseTool
from pydantic import BaseModel

from agent.src.core.tools import AddToCartInput
from agent.src.typedefs import EngineConfigName, EngineParams
from common.messages import RawMessageExchange
from experiments.config import ExperimentData, ExperimentId

ProviderBatchRequest = NewType("ProviderBatchRequest", dict[str, Any])
ProviderBatchResult = NewType("ProviderBatchResult", dict[str, Any])
ProviderBatchId = NewType("ProviderBatchId", str)


class BatchStatus(StrEnum):
    """Common, provider-agnostic batch status"""

    OUTSTANDING = "outstanding"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class ExperimentFailureModes(StrEnum):
    NO_TOOL_CALL = "No tool call found in response"
    API_ERROR = "Internal API error. See response_content"


class BatchRequest(BaseModel):
    """Common, provider-agnostic batch request

    This represents all experiments needed to be submitted to a provider.
    """

    # TODO: create strict `ExperimentMetadata` and `ExperimentProductData` objects
    experiments: list[ExperimentData]
    raw_messages: list[RawMessageExchange]
    engine_params: EngineParams
    tools: Optional[list[BaseTool]] = None

    class Config:
        arbitrary_types_allowed = True

    def tool_request_dict(self) -> list[dict[str, Any]] | None:
        if not self.tools:
            return None

        tools = []
        for tool in self.tools:
            tool_def = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.args_schema.model_json_schema(),
                },
            }
            tools.append(tool_def)
        return tools


class ExperimentResult(BaseModel):
    """Atomic unit of experiment result"""

    experiment_id: ExperimentId  # TODO: use `ExperimentMetadata` to condense
    response_content: str
    tool_call: Optional[AddToCartInput]
    failure_reason: Optional[ExperimentFailureModes] = None

    @property
    def success(self) -> bool:
        return self.failure_reason is None

    @property
    def failure(self) -> bool:
        return not self.success


class BatchResult(BaseModel):
    """Common, provider-agnostic batch result"""

    data: list[ExperimentResult]


class SerializedBatchRequest(BaseModel):
    """Wraps provider request with its associated experiment ID"""

    experiment_id: ExperimentId
    provider_request: ProviderBatchRequest


class ExperimentSubmissionRecord(BaseModel):
    """Links experiment tracking with provider batch monitoring"""

    experiment_id: ExperimentId
    batch_id: ProviderBatchId
    config_name: EngineConfigName
    failure_result: Optional[ExperimentResult] = None


class BatchStatusResult(BaseModel):
    """Simple result from batch monitoring containing just status and batch ID"""

    batch_id: ProviderBatchId
    status: BatchStatus
    result: Optional[ProviderBatchResult]


class BatchMonitorResult(BaseModel):
    """Result after monitor + deserializer coordination"""

    batch_id: ProviderBatchId
    experiment_ids: list[ExperimentId]
    status: BatchStatus
    result: Optional[ProviderBatchResult]
