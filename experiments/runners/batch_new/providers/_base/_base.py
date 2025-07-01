from abc import ABC

from experiments.config import EngineParams


class BaseBatchProvider(ABC):
    def __init__(self, engine_params: EngineParams):
        self.engine_params = engine_params
        self._setup()

    def _setup(self) -> None:
        pass
