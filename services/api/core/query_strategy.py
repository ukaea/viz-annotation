import numpy as np
from abc import ABC, abstractmethod

from services.api.schemas.samples import Sample


class QueryStrategy(ABC):
    def __init__(self, samples: list[Sample]):
        self.samples = samples

    @abstractmethod
    def next_sample_index(self):
        pass


class RandomQueryStrategy(QueryStrategy):
    def __init__(self, samples: list[Sample]):
        super().__init__(samples)
        self.indices = np.arange(len(self.samples))

    def next_sample_index(self) -> int:
        if len(self.indices) == 0:
            raise RuntimeError("No more samples to label!")

        index = np.random.choice(self.indices, replace=False, size=1)
        return self.indices.pop(index)


class SequentialQueryStrategy(QueryStrategy):
    def __init__(self, samples: list[Sample]):
        super().__init__(samples)
        self.index = 0

    def next_sample_index(self) -> int:
        if self.index > len(self.samples):
            raise RuntimeError("No more samples to label!")

        self.index += 1
        return self.index
