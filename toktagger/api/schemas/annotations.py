from typing import Literal, Optional, Union

from pydantic import Field, TypeAdapter, create_model, field_validator, model_validator

from toktagger.api.schemas import ConfiguredModel


class AnnotationBase(ConfiguredModel):
    """Base class for annotation inputs, without IDs."""

    label: str
    created_by: str
    validated: bool = False
    signal_name: Optional[str] = None
    uncertainty: Optional[float] = 1

    @model_validator(mode="before")
    def set_uncertainty(cls, values):
        if isinstance(values, dict):
            if "validated" in values and values["validated"]:
                values["uncertainty"] = 0
            elif values.get("uncertainty") is None:
                values["uncertainty"] = 1
        return values


class VideoAnnotationBase(AnnotationBase):
    frame: int
    track_id: str


class VideoFrameLabel(VideoAnnotationBase):
    type: Literal["video_frame_label"] = "video_frame_label"
    

class Annotation(AnnotationBase):
    id: str = Field(..., alias="_id")


class ClassLabel(AnnotationBase):
    type: Literal["class_label"] = "class_label"


class TimePoint(AnnotationBase):
    type: Literal["time_point"] = "time_point"
    time: float


class TimeRegion(AnnotationBase):
    type: Literal["time_region"] = "time_region"
    time_min: float
    time_max: float


class BoundingBox(AnnotationBase):
    type: Literal["bounding_box"] = "bounding_box"
    height: float
    width: float
    x_min: float
    y_min: float


class VideoBoundingBox(VideoAnnotationBase):
    type: Literal["video_bounding_box"] = "video_bounding_box"
    height: int
    width: int
    x_min: int
    y_min: int


class Polygon(AnnotationBase):
    type: Literal["polygon"] = "polygon"
    segmentation: list[list[float]] = Field(
        ...,
        min_length=1,
        max_length=1,
        description="COCO polygon segmentation as a list containing one polygon: [[x1, y1, x2, y2, ...]].",
    )

    @field_validator("segmentation")
    @classmethod
    def validate_segmentation(cls, v: list[list[float]]) -> list[list[float]]:
        for polygon in v:
            if len(polygon) < 6:
                raise ValueError("Each polygon must contain at least three points.")
            if len(polygon) % 2 != 0:
                raise ValueError(
                    "Each polygon must contain an even number of coordinates."
                )
        return v


class VideoPolygon(VideoAnnotationBase):
    type: Literal["video_polygon"] = "video_polygon"
    segmentation: list[list[int]] = Field(
        ...,
        min_length=1,
        max_length=1,
        description="COCO polygon segmentation as a list containing one polygon: [[x1, y1, x2, y2, ...]].",
    )

    @field_validator("segmentation")
    @classmethod
    def validate_segmentation(cls, v: list[list[int]]) -> list[list[int]]:
        for polygon in v:
            if len(polygon) < 6:
                raise ValueError("Each polygon must contain at least three points.")
            if len(polygon) % 2 != 0:
                raise ValueError(
                    "Each polygon must contain an even number of coordinates."
                )
        return v


class VideoPoint(VideoAnnotationBase):
    type: Literal["video_point"] = "video_point"
    x: int
    y: int


class AnnotationBatch(AnnotationBase):
    """Base class for batch annotation inputs, with or without IDs."""

    id: Optional[str] = Field(None, alias="_id")
    project_id: Optional[str] = None
    sample_id: Optional[str] = None
    shot_id: Optional[int] = None


class AnnotationOut(AnnotationBatch):
    """Base class for annotation outputs coming from the database, with IDs."""

    id: str = Field(alias="_id")
    project_id: str
    sample_id: str
    shot_id: int


def create_out_model(base_class, name_suffix="Out"):
    """Create an Out variant of an annotation class by combining it with AnnotationOut."""
    class_name = f"{base_class.__name__}{name_suffix}"
    return create_model(
        class_name,
        __base__=(base_class, AnnotationOut),
    )


def create_batch_model(base_class, name_suffix="Batch"):
    """Create a Batch variant of an annotation class by combining it with AnnotationBatch."""
    class_name = f"{base_class.__name__}{name_suffix}"
    return create_model(
        class_name,
        __base__=(base_class, AnnotationBatch),
    )


# Generate Out classes using factory function
TimePointOut = create_out_model(TimePoint)
TimeRegionOut = create_out_model(TimeRegion)
BoundingBoxOut = create_out_model(BoundingBox)
PolygonOut = create_out_model(Polygon)
VideoBoundingBoxOut = create_out_model(VideoBoundingBox)
VideoPolygonOut = create_out_model(VideoPolygon)
VideoPointOut = create_out_model(VideoPoint)
VideoFrameLabelOut = create_out_model(VideoFrameLabel)
ClassLabelOut = create_out_model(ClassLabel)

# Generate Batch classes using factory function
TimePointBatch = create_batch_model(TimePoint)
TimeRegionBatch = create_batch_model(TimeRegion)
BoundingBoxBatch = create_batch_model(BoundingBox)
PolygonBatch = create_batch_model(Polygon)
VideoBoundingBoxBatch = create_batch_model(VideoBoundingBox)
VideoPolygonBatch = create_batch_model(VideoPolygon)
VideoPointBatch = create_batch_model(VideoPoint)
VideoFrameLabelBatch = create_batch_model(VideoFrameLabel)
ClassLabelBatch = create_batch_model(ClassLabel)


# Union types for annotations
AnnotationTypes = Union[
    TimePoint,
    TimeRegion,
    BoundingBox,
    Polygon,
    VideoBoundingBox,
    VideoPolygon,
    VideoPoint,
    VideoFrameLabel,
    ClassLabel,
]

AnnotationOutTypes = Union[
    TimePointOut,
    TimeRegionOut,
    BoundingBoxOut,
    PolygonOut,
    VideoBoundingBoxOut,
    VideoPolygonOut,
    VideoPointOut,
    VideoFrameLabelOut,
    ClassLabelOut,
]

AnnotationBatchTypes = Union[
    TimePointBatch,
    TimeRegionBatch,
    BoundingBoxBatch,
    PolygonBatch,
    VideoBoundingBoxBatch,
    VideoPolygonBatch,
    VideoPointBatch,
    VideoFrameLabelBatch,
    ClassLabelBatch,
]

# TypeAdapters for annotations
AnnotationTypeAdapter = TypeAdapter(AnnotationTypes)
AnnotationOutTypeAdapter = TypeAdapter(AnnotationOutTypes)
AnnotationBatchTypeAdapter = TypeAdapter(AnnotationBatchTypes)
