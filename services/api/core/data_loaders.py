import pandas as pd
from abc import ABC, abstractmethod

from services.api.schemas.samples import Sample


class DataLoader(ABC):
    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, index):
        pass


class ParquetDataLoader(DataLoader):
    def __init__(self, samples: list[Sample]):
        self.file_names = [sample.file_name for sample in samples]

    def __len__(self) -> int:
        return len(self.file_names)

    def __getitem__(self, index) -> list[dict[str, list]]:
        df = pd.read_parquet(self.file_names[index])
        data = df.to_dict("records")
        return data
