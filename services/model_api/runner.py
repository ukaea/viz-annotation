from model import model_registry
from models.elm_model.main import ELMModel
from db_client import DBClient
from multiprocessing import Process

model_registry.register('elms', ELMModel)
class ModelRunner:
    def __init__(self):
        pass

    def run(self, name: str):
        process = Process(target=self._do_run, args=(name,))
        process.start()
        
    def _do_run(self, name:str):
        db = DBClient()
        items = db.get_annotations()
        model = model_registry.create(name)
        model.run(items)