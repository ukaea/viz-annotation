from toktagger.api.schemas.samples import Sample
from toktagger.api.schemas.annotations import Annotation, VideoBoundingBox
import pydantic
from toktagger.api.models.base import Model, ModelRegistry
from toktagger.api.schemas.data import ImageParams
import pathlib


class VideoCNNTrainParams(pydantic.BaseModel):
    num_epochs: int


class VideoCNNPredictParams(pydantic.BaseModel):
    current_frame: bool = pydantic.Field(
        default=False,
        description="Only applicable within sample predictions - whether to predict on the current frame, or the whole sample.",
    )


@ModelRegistry.register(
    "video-cnn", ["video"], VideoCNNTrainParams, VideoCNNPredictParams
)
class VideoCNN(Model):
    def define_model(self):
        return None

    def train(
        self,
        samples: list[Sample],
        annotations: list[list[Annotation]],
        params: VideoCNNTrainParams,
    ):
        return params.num_epochs

    def predict(self, samples, params: VideoCNNPredictParams, data_params: ImageParams):
        annotations = []
        for sample in samples:
            if params.current_frame:
                data = self.data_loader.get_sample(sample, data_params)
                annotations.append(
                    [
                        VideoBoundingBox(
                            label="UFO",
                            height=50,
                            width=50,
                            x_min=0,
                            y_min=0,
                            frame=data.frame,
                            created_by=f"model::{self.type}",
                            track_id="test",
                        )
                    ]
                )
            else:
                # Get first frame
                data = self.data_loader.get_sample(
                    sample, ImageParams(name="image", frame=None)
                )
                sample_anns = [
                    VideoBoundingBox(
                        label="UFO",
                        height=50,
                        width=50,
                        x_min=0,
                        y_min=0,
                        frame=data.frame,
                        created_by=f"model::{self.type}",
                        track_id="test",
                    )
                ]
                # keep going until no more files
                while True:
                    try:
                        data = self.data_loader.get_sample(
                            sample, ImageParams(name="image", frame=data.frame + 1)
                        )
                        sample_anns.append(
                            VideoBoundingBox(
                                label="UFO",
                                height=50,
                                width=50,
                                x_min=0,
                                y_min=0,
                                frame=data.frame,
                                created_by=f"model::{self.type}",
                                track_id="test",
                            )
                        )

                    except FileNotFoundError:
                        break
                annotations.append(sample_anns)
        return annotations

    def save(self, file_stem: str):
        pathlib.Path(file_stem).with_suffix(".model").touch()

    def load(self, file_path: str):
        self.model = None
