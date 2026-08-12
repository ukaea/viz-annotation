from tensorflow.keras.callbacks import Callback
from toktagger.api.core.sender import send_model_updates
from toktagger.api.schemas.models import ModelUpdate


class ModelProgress(Callback):
    def __init__(
        self, project_id: str, model_id: str, num_epochs: int, eval_metric: str
    ):
        self.project_id = project_id
        self.model_id = model_id
        self.num_epochs = num_epochs
        self.eval_metric = eval_metric
        super().__init__()

    def on_train_begin(self):
        model_update = ModelUpdate(status="training", progress=0)
        send_model_updates(
            project_id=self.project_id, model_id=self.model_id, updates=model_update
        )

    def on_epoch_end(self, epoch: int, logs: dict):
        model_update = ModelUpdate(
            progress=(epoch / self.num_epochs) * 100,
            score=logs.get(self.eval_metric),
        )
        send_model_updates(
            project_id=self.project_id, model_id=self.model_id, updates=model_update
        )

    def on_train_end(self, logs: dict):
        model_update = ModelUpdate(
            status="completed",
            progress=100,
            score=logs.get(self.eval_metric),
        )
        send_model_updates(
            project_id=self.project_id, model_id=self.model_id, updates=model_update
        )
