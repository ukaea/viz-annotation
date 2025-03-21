from enum import Enum
from abc import ABC, abstractmethod


class AnnotatorType(str, Enum):  # noqa: F821
    CLASSIC = "classic"
    UNET = "unet"


class DataAnnotator(ABC):
    @abstractmethod
    def get_annotations(self, shot_id: int, **kwargs):
        pass

    @abstractmethod
    def train(self, shot_ids: list[int], annotations):
        pass

    @abstractmethod
    def evaluate(self, shot_ids: list[int], annotations):
        pass

    @abstractmethod
    def score(self, shot_ids: list[int]):
        pass

    def save_model(self):
        pass

    def load_model(self):
        pass
