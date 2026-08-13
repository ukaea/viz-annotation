from datetime import datetime

from bson.errors import InvalidId
from bson.objectid import ObjectId
from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ConfiguredModel(BaseModel):
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Time when this object was created, leave blank to automatically generate.",
    )

    @model_validator(mode="before")
    def convert_objectid(cls, values):
        # Only runs against a raw dict (e.g. incoming request JSON or a DB
        # document) - an already-constructed model instance (e.g. during response
        # serialization) is left untouched, same as before this narrowed from a
        # bare `key in values` check.
        if isinstance(values, dict):
            for key in ("_id", "project_id", "sample_id"):
                # Only stringify a real value - str(None) is the string "None",
                # which would corrupt an explicit null into a truthy value on
                # fields like AnnotationBatch.id, where callers rely on it being
                # None to mean "no id yet".
                if values.get(key) is not None:
                    values[key] = str(values[key])
        return values

    model_config = ConfigDict(
        use_enum_values=True, json_encoders={ObjectId: str}, validate_by_name=True
    )


def convert_to_objectid(id: str, collection: str):
    try:
        obj_id = ObjectId(id)
    except InvalidId:
        raise HTTPException(
            status_code=400, detail=f"{collection[:-1].title()} ID is not valid."
        )
    return obj_id
