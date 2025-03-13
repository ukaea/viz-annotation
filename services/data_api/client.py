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
        await self.collection.insert_one(item.model_dump())

    async def upsert(self, shot: Shot):
        await self.collection.update_one(
            {"shot_id": shot.shot_id}, {"$set": shot.model_dump()}, upsert=True
        )

    async def list(self):
        items_cursor = self.collection.find()
        items = await items_cursor.to_list()
        return items

    async def find(self, shot_id: int) -> Shot:
        item = await self.collection.find_one({"shot_id": int(shot_id)})
        item.pop("_id", None)
        return Shot(**item)

    async def delete(self, shot_id: int):
        delete_result = await self.collection.delete_one({"shot_id": shot_id})
        if delete_result.deleted_count == 0:
            raise RuntimeError(detail="Item not found")
        return {"message": "Item deleted successfully"}
