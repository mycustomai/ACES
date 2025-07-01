from abc import abstractmethod

from experiments.runners.batch_new.typedefs import (BatchResult,
                                                    ProviderBatchResult)

from ._base import BaseBatchProvider


class BaseBatchProviderDeserializer(BaseBatchProvider):
    """Functor for converting provider-specific batch results into a common batch result"""

    @abstractmethod
    def deserialize(self, data: ProviderBatchResult) -> BatchResult: ...
