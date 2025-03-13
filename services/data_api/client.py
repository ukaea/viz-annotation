import os
from motor.motor_asyncio import AsyncIOMotorClient
from model import Shot

# MongoDB Connection URL
MONGO_URL = os.environ["MONGO_URL"]
DATABASE_NAME = "event_db"
COLLECTION_NAME = "shots"


class MongoDBClient:
    def __init__(self, collection_name: str = COLLECTION_NAME):
        self.client = AsyncIOMotorClient(MONGO_URL)
        self.db = self.client[DATABASE_NAME]
        self.collection = self.db[collection_name]

    async def insert(self, item: Shot):
        new_item = await self.collection.insert_one(item.model_dump())
        return {"id": str(new_item.inserted_id), **item.model_dump()}

    async def upsert(self, shot: Shot):
        updated_result = await self.collection.update_one(
            {"shot_id": shot.shot_id}, {"$set": shot.model_dump()}, upsert=True
        )
        return {"id": str(updated_result.upserted_id), **shot.model_dump()}

    async def list(self):
        items_cursor = self.collection.find()
        items = await items_cursor.to_list()
        return [{"id": str(item["_id"]), **item} for item in items]

    async def find(self, shot_id: int):
        item = await self.collection.find_one({"shot_id": shot_id})
        if not item:
            return None
        return {"id": str(item["_id"]), **item}

    async def delete(self, shot_id: int):
        delete_result = await self.collection.delete_one({"shot_id": shot_id})
        if delete_result.deleted_count == 0:
            raise RuntimeError(detail="Item not found")
        return {"message": "Item deleted successfully"}
