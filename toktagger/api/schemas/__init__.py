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
        for key in ("_id", "project_id", "sample_id"):
            if key in values:
                values[key] = str(values.get(key))
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
