from abc import abstractmethod

from experiments.runners.batch_runtime.typedefs import (BatchStatusResult,
                                                        ProviderBatchId)

from ._base import BaseBatchProvider


class BaseBatchProviderMonitor(BaseBatchProvider):
    @abstractmethod
    def monitor_batches(
        self, batches: list[ProviderBatchId]
    ) -> list[BatchStatusResult]:
        """
        Monitor multiple batches simultaneously and return results for any that have completed.

        This method is designed for bulk monitoring efficiency - it checks many batches
        in a single API call and returns raw results only for batches that have actually
        completed or failed.

        Args:
            batches: List of batch IDs to monitor

        Returns:
            List of BatchStatusResult objects, one for each batch that has completed
            or has new results available. Empty list if no batches are ready.
        """
        ...
