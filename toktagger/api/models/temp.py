from toktagger.api.schemas.samples import Sample
from toktagger.api.schemas.annotations import Annotation, VideoBoundingBox
import pydantic
from toktagger.api.models.base import Model, ModelRegistry
from toktagger.api.schemas.data import ImageParams
import pathlib

from ultralytics import YOLO

import base64
from io import BytesIO
from PIL import Image


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
                            created_by="video-cnn",
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
                        created_by="video-cnn",
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
                                created_by="video-cnn",
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


@ModelRegistry.register(
    "YOLOv8",
    ["video"],
)
class YOLOv8(Model):
    def define_model(self) -> YOLO:
        return YOLO("yolov8n")

    def train(
        self,
        samples: list[Sample],
        annotations: list[list[Annotation]],
        params: None,
    ):
        return None

    def predict(self, samples, params: VideoCNNPredictParams, data_params: ImageParams):
        annotations = []
        data = None
        for sample in samples:
            anns = []
            # keep going until no more files
            while True:
                if not data:
                    # Get first frame
                    data = self.data_loader.get_sample(
                        sample, ImageParams(name="image", frame=None)
                    )
                else:
                    try:
                        data = self.data_loader.get_sample(
                            sample, ImageParams(name="image", frame=data.frame + 1)
                        )
                    except FileNotFoundError:
                        break
                image = Image.open(BytesIO(base64.b64decode(data.values)))

                result = self.model.predict(
                    image,
                    verbose=False,
                )[0]

                for box in result.boxes:
                    xyxy = box.xyxy[0].cpu().numpy()
                    anns.append(
                        VideoBoundingBox(
                            label="UFO",
                            height=int(xyxy[3] - xyxy[1]),
                            width=int(xyxy[2] - xyxy[0]),
                            x_min=int(xyxy[0]),
                            y_min=int(xyxy[1]),
                            frame=data.frame,
                            created_by="YOLOv8",
                            track_id="test",
                        )
                    )

            annotations.append(anns)

        return annotations

    def save(self, file_stem: str) -> None:
        output = f"{file_stem}.pt"
        self.model.save(output)

    def load(self, file_path: str) -> None:
        self.model = YOLO(file_path)
