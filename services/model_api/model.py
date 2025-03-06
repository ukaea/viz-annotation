from abc import ABC, abstractmethod

class Model(ABC):

    @abstractmethod
    def train(self, annotations):
        """Abstract method that must be implemented to run the model"""
        pass

    @abstractmethod
    def query(self):
        """Abstract method that must be implemented to run the model"""
        pass