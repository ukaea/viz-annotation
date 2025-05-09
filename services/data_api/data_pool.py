from pathlib import Path
from typing import List
import numpy as np
import pandas as pd


class DataPool:
    def __init__(self, data_path: str):
        self.batch_size = 3
        file_names = Path(data_path).glob("*.parquet")
        self.shots = [name.stem for name in file_names]
        self.training_future = None
        self.validated_shots = []
        self.scores = []

    @property
    def pool_size(self) -> int:
        return len(self.shots)

    @property
    def num_validated(self) -> int:
        return len(self.validated_shots)

    @property
    def unlabelled_shots(self):
        return [shot for shot in self.shots if shot not in self.validated_shots]

    @property
    def currently_training(self) -> bool:
        return self.training_future is not None and not self.training_future.ready()

    def retrain(self) -> bool:
        # Check if we have enough data to retrain
        if self.num_validated % self.batch_size != 0:
            return False

        # Check if we have any validated shots yet
        if self.num_validated == 0:
            return False

        # Check if we are currently training
        if self.currently_training:
            return False

        return True

    def set_validated(self, shot_ids: List[int]):
        self.validated_shots = shot_ids

    def query(self) -> int:
        return int(np.random.choice(self.unlabelled_shots))
