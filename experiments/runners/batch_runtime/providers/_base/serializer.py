from abc import abstractmethod

from experiments.runners.batch_runtime.typedefs import (BatchRequest,
                                                        SerializedBatchRequest)

from ._base import BaseBatchProvider


class BaseBatchProviderSerializer(BaseBatchProvider):
    """Converts common batch types to provider-specific types"""

    @abstractmethod
    def serialize(self, data: BatchRequest) -> list[SerializedBatchRequest]: ...
