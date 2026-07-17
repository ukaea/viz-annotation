from __future__ import annotations

import logging

import pydantic

from toktagger.api.models.base import Model, ModelRegistry
from toktagger.api.schemas.annotations import Annotation
# from toktagger.api.schemas.data import DataParams # not useful for video data
from toktagger.api.schemas.data import ImageParams
from toktagger.api.schemas.samples import Sample

from ultralytics import YOLO
# convert the images from base64 to ultralytics usable format
import base64
from io import BytesIO
from PIL import Image



logger = logging.getLogger(__name__)

class DebugVideoTrainParams(pydantic.BaseModel):
    max_samples: int = 1


class DebugVideoPredictParams(pydantic.BaseModel):
    max_samples: int = 1

def yolo_forward_pass(image_string):
    """
    Yolo naive inference on a image from toktagger.
    The pipeline is base64 string → bytes → PIL image → YOLO
    """
    model = YOLO("yolo26n.pt")
    b64_string = image_string
    img_bytes = base64.b64decode(b64_string)
    img = Image.open(BytesIO(img_bytes)).convert("RGB")

    results = model(img)
    print(results)


# def build_video_frame_manifest(
#     samples: list[Sample],
#     annotations: list[list[Annotation]],
#     class_map: dict[str, int],
# ) -> list[dict]:
#     """
#     Convert TokTagger video samples into image-level detection records.
#     One TokTagger sample = one shot directory.
#     One record = one frame image.
#     One frame can contain multiple bounding boxes.
#     Inputs:
#     * list of Samples this the entire shot with multiple frams in each sample.
#     *list of (list of annotations) can be accessed using annotations[shot_id][frame_num].
#     """
#     if len(samples) != len(annotations):
#         raise ValueError("Samples and annotations must have the same length")

#     frame_manifest = []

#     for sample, ann_list in zip(samples, annotations):
#         shot_id = int(sample.shot_id)
#         shot_dir = Path(sample.data.file_name)

#         if not shot_dir.exists():
#             logger.warning("Shot directory does not exist: %s", shot_dir)
#             continue

#         grouped = {}

#         for ann in ann_list:
#             if getattr(ann, "type", None) != "video_bounding_box":
#                 continue
#             if not getattr(ann, "validated", False):
#                 continue

#             frame = int(ann.frame)
#             grouped.setdefault(frame, []).append(ann)

#         for image_path in sorted(shot_dir.glob("*.png")):
#             try:
#                 frame = int(image_path.stem)
#             except ValueError:
#                 continue

#             frame_annotations = grouped.get(frame, [])

#             boxes = []
#             classes = []
#             labels = []
#             track_ids = []

#             for ann in frame_annotations:
#                 if ann.label not in class_map:
#                     logger.warning("Unknown label skipped: %s", ann.label)
#                     continue

#                 x1 = float(ann.x_min)
#                 y1 = float(ann.y_min)
#                 x2 = x1 + float(ann.width)
#                 y2 = y1 + float(ann.height)

#                 boxes.append([x1, y1, x2, y2])
#                 classes.append(class_map[ann.label])
#                 labels.append(ann.label)
#                 track_ids.append(getattr(ann, "track_id", None))

#             frame_manifest.append(
#                 {
#                     "shot_id": shot_id,
#                     "frame": frame,
#                     "image_path": str(image_path),
#                     "boxes": boxes,
#                     "classes": classes,
#                     "labels": labels,
#                     "track_ids": track_ids,
#                 }
#             )

#     return frame_manifest


@ModelRegistry.register(
    "debug_video_get_sample_no_training",
    ["video"],
    DebugVideoTrainParams,
    DebugVideoPredictParams,
)
class DebugVideoGetSampleModel(Model):
    def __init__(self, model_id: str, project) -> None:
        super().__init__(model_id=model_id, project=project)

        self.model_id = model_id
        self.project = project
        self.type = "debug_video_get_sample"
        self._trained = True

    def define_model(self):
        return "debug_video_get_sample_placeholder"
    def get_sample_image(self, sample):
        """
        Get a base64 image from toktagger for a single frame.
        Do the same for the entire sample and return the manifest file.
        This will then be used by the manifest list[dict].
        The main for-loop to loop over the samples will run in the manifest builder functions, not here.
        The manifest should use the self.data_loader and then we use it to get the samples as  self.data_loader.get_sample()
        """

        sample_manifest = [] # a list[dict] type object
        # Get first frame
        frame_image = self.data_loader.get_sample(
            sample, ImageParams(name="image", frame=None)
        )
        # add the first frame to the sample_manifest
        sample_manifest.append({
                "shot_id": sample.shot_id,
                "frame": frame_image.frame, # frame number
                "image": frame_image.values # base64 encode image
        })

        while True:
            try:
                frame_image = self.data_loader.get_sample(
                    sample,
                    ImageParams(name="image", frame = frame_image.frame + 1 )
                )
                # print("Frame number: ",frame_image.frame)
                sample_manifest.append({
                    "shot_id": sample.shot_id,
                    "frame": frame_image.frame,
                    "image": frame_image.values
                })

            except FileNotFoundError:
                # print("Can't access the sample. Check with toktagger devs. Read the following error.")
                print("----------------------")
                print("End of frames I guess")
                print("last frame was: ", frame_image.frame)
                print("----------------------")
                break #once
        return sample_manifest



    def naive_manifest(
            self,
            samples: list[Sample]) -> list[dict]:
        from time import perf_counter # calcualete elapsed time
        full_manifest = [] # list where we store all the frame wise dicts that we will eventually return.
        for sample in samples:
            """
            A single sample will return a datetime instance
            timestamp=datetime.datetime(
            2026, 7, 7, 10, 43, 51, 957000) 
            shot_id=104521 
            data=ImageFileData(file_name='/Users/zw5893/Desktop/repos/toktagger_dev/data/JET_UFO/images/104521', 
            protocol='file', type='png') 
            validated_annotations=True 
            id='6a4cca57e323ae28f34b129a' 
            project_id='6a4cca57e323ae28f34b1297'
            We really care about shot_id, frame_num and the sample_frame_images
            """
    
            print("sample: ",sample)
            start = perf_counter() # start timer
            sample_manifest = self.get_sample_image(sample) # base64 encoded image
            elapsed = perf_counter() - start
            print("Sample manifest shape: ", len(sample_manifest))
            # timer related metrics
            print(f"Frames: {len(sample_manifest)}")
            print(f"Time: {elapsed:.3f} s")
            print(f"Average: {elapsed / len(sample_manifest):.4f} s/frame")
            print("-" * 50)
            full_manifest.append({
                ############## checkpoint
                sample.shot_id: sample_manifest # each sample in the fill manifest will be keyed with its shot_id
                }) # so full_manifest[sample_number][frame_wise_manifest]
        print("Full manifest shape (same as number of samples)): ", len(full_manifest))
        print("######################################")

        # print(full_manifest[0]) # don't print very logn output


    def train(
        self,
        samples: list[Sample],
        annotations: list[list[Annotation]],
        params: DebugVideoTrainParams | None = None,
    ) -> float:
        self.log_progress(training_status="started")
        if params is None:
            params = DebugVideoTrainParams()

        print("################ DEBUG VIDEO GET_SAMPLE TRAIN ################")
        print("num samples:", len(samples))
        print("Num annotations: ", len(annotations))
        print("data_loader:", type(self.data_loader), self.data_loader)
        print("print params: ", params)
        self.naive_manifest(samples, )
        for i, sample in enumerate(samples):
            print("--------------- SAMPLE", i, "---------------")

            frame = None
            if annotations and annotations[i]:
                frame = annotations[i][0].frame
                # print("################ \n bbox: \n ################ \n", type(annotations[i][0]))
                print("Annotation sample length (number of frames): ", len(annotations[i]))
                # annotation number 20 of each shot
                print("Annotation frame number: ", annotations[i][20].frame)

                """
                The challenge is. annotations and sample are separate. We have to create the yolo manifest
                by somehow working together with the image and data.
                so images can be access as data.values in base64 format
                and annotations can be accessed as annotations[shot_id][frame_num]
                not all frame_nume will have an annotations.
                This needs a function to do a mapping of image and its annotations
                then put them into manifest dictionary.

                The manifest could look like this:
                {'shot_id': 104520, 'frame': 300, 'image_path': 'images/104520/300.png', 'boxes': [], 'classes': [], 'labels': [], 'track_ids': []}
                {'shot_id': 104520, 'frame': 301, 'image_path': 'images/104520/301.png', 'boxes': [], 'classes': [], 'labels': [], 'track_ids': []}
                """




            try:
                # give individual frames
                # unless we put it in a while True loop we get the first frame of the shot
                data = self.data_loader.get_sample(
                    sample,
                    ImageParams(name="image", frame=frame),
                )

                # print("get_sample returned:", type(data))
                # print("sample_debug type:", type(data.values))
                print("Number of frames:", data.frame)
                # yolo_forward_pass(data.values) # yolo forward pass workign so comment it for now.


            except Exception as exc:
                print("get_sample FAILED:", repr(exc))

        self.log_progress(training_status="completed")
        return 0.0

    def predict(self, samples, params=None, data_params=None):
        if params.current_frame:
            logger.info("Predict called for one frame")
        return [[] for _ in samples]

    def save(self, file_stem: str):
        pass

    def load(self, file_path: str):
        pass