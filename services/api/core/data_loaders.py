import pandas as pd
import pyuda
from abc import ABC, abstractmethod

from services.api.schemas.samples import FileData, Sample, ShotData


class DataLoader(ABC):
    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, index):
        pass


class ParquetDataLoader(DataLoader):
    """DataLoader for retrieving data using a folder of Parquet files"""

    def __init__(self, samples: list[Sample]):
        self.data_items: list[FileData] = [sample.data for sample in samples]

    def __len__(self) -> int:
        return len(self.data_items)

    def __getitem__(self, index) -> list[dict[str, list]]:
        item: FileData = self.data_items[index]
        df = pd.read_parquet(item.file_name)
        df = df[item.column_names]
        data = df.to_dict("records")
        return data


class UDADataLoader(DataLoader):
    """DataLoader for retrieving data using the UDA access layer"""

    def __init__(self, samples: list[Sample]):
        self.client = pyuda.Client()
        self.data_items: list[ShotData] = [sample.data for sample in samples]

    def __len__(self) -> int:
        return len(self.data_items)

    def __getitem__(self, index):
        item: ShotData = self.data_items[index]

        results = {}
        for name in item.signal_names:
            signal = self.client.get(item.shot_id, name)
            results[name] = signal.data

        return results
