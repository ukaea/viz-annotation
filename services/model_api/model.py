from abc import ABC, abstractmethod
from registry import Registry

class Model(ABC):

    @abstractmethod
    def run(self, annotations):
        """Abstract method that must be implemented to run the model"""
        pass

model_registry = Registry[Model]()