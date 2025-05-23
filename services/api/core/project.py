from typing import List
from services.api.schemas.projects import (
    DataLoaderType,
    Project as ProjectMetadata,
    QueryStrategyType,
)
from services.api.core.query_strategy import (
    QueryStrategy,
    RandomQueryStrategy,
    SequentialQueryStrategy,
)
from services.api.core.data_loaders import DataLoader, ParquetDataLoader, UDADataLoader
from services.api.core.data_pool import DataPool
from services.api.schemas.samples import Sample


class Project:
    def __init__(self, metadata: ProjectMetadata):
        self.name = metadata.name
        self.samples = metadata.samples
        data_loader = self.get_data_loader(self.samples, metadata.data_loader)
        query_strategy = self.get_query_strategy(self.samples, metadata.query_strategy)
        self.data_pool = DataPool(data_loader, query_strategy)

    def get_next_sample(self):
        data_item = self.data_pool.get_next_sample()
        return data_item

    def set_data_pool(self):
        self.data_pool = DataPool(self.data_loader, self.query_strategy)

    def get_data_loader(
        self, samples: List[Sample], data_loader_type: DataLoaderType
    ) -> DataLoader:
        if data_loader_type == DataLoaderType.PARQUET:
            data_loader = ParquetDataLoader(samples)
        elif data_loader_type == DataLoaderType.UDA:
            data_loader = UDADataLoader(samples)
        else:
            raise RuntimeError(f"Unknown query strategy {data_loader_type}")

        return data_loader

    def get_query_strategy(
        self, samples: List[Sample], strategy: QueryStrategyType
    ) -> QueryStrategy:
        if strategy == QueryStrategyType.SEQUENTIAL:
            qs = SequentialQueryStrategy(samples)
        elif strategy == QueryStrategyType.RANDOM:
            qs = RandomQueryStrategy(samples)
        else:
            raise RuntimeError(f"Unknown query strategy {strategy}")

        return qs
