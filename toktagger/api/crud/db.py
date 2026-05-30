from pathlib import Path
import pymongo
import pydantic
import typing
from bson.objectid import ObjectId

from platformdirs import user_cache_dir
from toktagger.api.crud.mongita_client import AsyncMongitaClient

DATABASE_NAME = "event_db"
COLLECTION_NAME = "shots"

T = typing.TypeVar("T", bound=pydantic.BaseModel)


class MongoDBClient:
    def __init__(self, url: str, db_name: str, cache_dir: str | None = None):
        if url.startswith("mongodb://"):
            # Use mongodb (expects running instance of mongodb at this address)
            self.client = pymongo.AsyncMongoClient(url)
        else:
            if not cache_dir:
                cache_dir = user_cache_dir("toktagger", "ukaea")
            cache_dir = Path(cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            file_name = cache_dir / db_name
            self.client = AsyncMongitaClient(file_name)
        self.db = self.client[db_name]

    async def insert(
        self,
        collection: typing.Literal["projects", "annotations", "models", "samples", "users", "project_members"],
        model: T,
        ids: dict[str, ObjectId] | None = None,
    ):
        ids = ids or {}
        document = model.model_dump(mode="python")
        document.update(ids)
        result = await self.db[collection].insert_one(document)
        return str(result.inserted_id)

    async def insert_many(
        self,
        collection: typing.Literal["projects", "annotations", "models", "samples", "users", "project_members"],
        models: list[T],
        ids: typing.Union[dict, list[dict]] | None = None,
    ):
        ids = ids or {}
        documents = [model.model_dump(mode="python") for model in models]

        if type(ids) is list:
            if len(ids) != len(models):
                raise ValueError(
                    "If providing IDs as a list, must be the same length as models"
                )
            documents = [{**document, **_id} for document, _id in zip(documents, ids)]
        else:
            documents = [{**document, **ids} for document in documents]

        result = await self.db[collection].insert_many(documents)
        return [str(object_id) for object_id in result.inserted_ids]

    async def update(
        self,
        collection: typing.Literal["projects", "annotations", "models", "samples", "users", "project_members"],
        model: T,
        object_id: ObjectId,
    ):
        # Retrieve existing entry:
        document = await self.db[collection].find_one({"_id": object_id})

        # Add updates to db entry
        updated_document = {
            **document,
            **model.model_dump(mode="python", exclude_unset=True, exclude_none=True),
        }

        return await self.db[collection].update_one(
            {"_id": object_id}, {"$set": updated_document}
        )

    async def get_document_by_id(
        self,
        collection: typing.Literal["projects", "annotations", "models", "samples", "users", "project_members"],
        object_id: ObjectId,
    ):
        return await self.db[collection].find_one({"_id": object_id})

    async def get_all_documents(
        self, collection: typing.Literal["projects", "annotations", "models", "samples"]
    ):
        all_documents = self.db[collection].find()
        return await all_documents.to_list()

    async def get_filtered_documents(
        self,
        collection: typing.Literal["projects", "annotations", "models", "samples", "users", "project_members"],
        filters: dict = {},
        sort_by: str = "_id",
        sort_direction: typing.Literal["ascending", "descending"] = "descending",
        start=0,
        limit=0,
    ):
        direction = (
            pymongo.ASCENDING if sort_direction == "ascending" else pymongo.DESCENDING
        )
        documents = self.db[collection].find(
            filters,
            sort=[(sort_by, direction)],
            skip=start,
            limit=limit,
        )
        return await documents.to_list()

    async def delete_filtered_documents(
        self,
        collection: typing.Literal["projects", "annotations", "models", "samples", "users", "project_members"],
        filters: dict = {},
    ):
        return await self.db[collection].delete_many(filters)


# Notes to self
# I am planning on doing a collection per route
# Eg Projects, Annotations, Samples, Models etc etc
# These would then be linked via IDs
# So each annotation document would have a 'sample_id' and a 'project_id' for example
# This means that you can search through them using that ID
# Eg above, get_documents_per_project would get you all documents from a collection which are relevant for a certain project
