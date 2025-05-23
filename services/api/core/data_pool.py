from services.api.core.data_loaders import DataLoader
from services.api.core.query_strategy import QueryStrategy


class DataPool:
    def __init__(self, data_loader: DataLoader, query_strategy: QueryStrategy):
        self.data_loader = data_loader
        self.query_strategy = query_strategy

    def get_next_sample(self):
        index = self.query_strategy.next_sample_index()
        return self.data_loader[index]
