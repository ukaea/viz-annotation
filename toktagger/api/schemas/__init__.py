from datetime import datetime, timezone
from pydantic import BaseModel, Field, model_validator, ConfigDict, field_serializer
from bson.objectid import ObjectId
from bson.errors import InvalidId
from fastapi import HTTPException


class ConfiguredModel(BaseModel):
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Time when this object was created, leave blank to automatically generate.",
    )

    @field_serializer("timestamp")
    def serialize_timestamp(self, value: datetime) -> str:
        """Ensure RFC 3339 compliant date-time format with timezone."""
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.isoformat()

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
