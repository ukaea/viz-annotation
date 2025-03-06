from model import Model, model_registry
from models.elm_model.main import ELMModel
from db_client import DBClient
from multiprocessing import Process

class ModelWorker:
    
    def __init__(self, model: Model):
        self.model = model
        self.db = DBClient()

    def __call__(self):
        items = self.db.get_annotations()
        try:
            self.model.run(items)
        except Exception as e:
            import traceback
            print(traceback.format_exc())

class ModelRunner:
    def __init__(self):
        self.models = {
            'elms': ELMModel()
        }

    def run(self, name: str):
        worker = ModelWorker(self.models[name])
        process = Process(target=worker)
        process.start()

    def query(self, name: str) -> int:
        return self.models[name].query()