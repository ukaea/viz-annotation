from services.api.schemas.projects import Project as ProjectMetadata
from services.api.core.query_strategy import SequentialQueryStrategy
from services.api.core.data_loaders import ParquetDataLoader
from services.api.core.data_pool import DataPool


class Project:
    def __init__(self, project: ProjectMetadata):
        self.name = project.name

        query_strategy = SequentialQueryStrategy(project.samples)
        data_loader = ParquetDataLoader(project.samples)
        self.data_pool = DataPool(data_loader, query_strategy)

    def get_next_sample(self):
        data_item = self.data_pool.get_next_sample()
        return data_item
