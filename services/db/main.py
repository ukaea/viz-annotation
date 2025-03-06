from fastapi import FastAPI, HTTPException, Depends
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel
from typing import List, Optional
from model import Shot, ShotInDB

# MongoDB Connection URL
MONGO_URL = "mongodb://root:example@mongodb:27017"
DATABASE_NAME = "event_db"
COLLECTION_NAME = "shots"

# Initialize FastAPI app
app = FastAPI()

# MongoDB Client
client = AsyncIOMotorClient(MONGO_URL)
db = client[DATABASE_NAME]
collection = db[COLLECTION_NAME]


@app.post("/shots/", response_model=ShotInDB)
async def create_item(item: Shot):
    print(item)
    new_item = await collection.insert_one(item.model_dump())
    return {"id": str(new_item.inserted_id), **item.model_dump()}


@app.get("/shots/", response_model=List[ShotInDB])
async def get_items():
    items_cursor = collection.find()
    items = await items_cursor.to_list(length=100)
    return [{"id": str(item["_id"]), **item} for item in items]


@app.get("/shots/{shot_id}", response_model=ShotInDB)
async def get_item(shot_id: str):
    item = await collection.find_one({"shot_id": shot_id})
    if item:
        return {"id": str(item["_id"]), **item}
    raise HTTPException(status_code=404, detail="Item not found")


@app.put("/shots/{shot_id}", response_model=ShotInDB)
async def update_item(item_id: str, shot: Shot):
    updated_result = await collection.update_one(
        {"_shot_id": item_id}, {"$set": shot.model_dump()}
    )
    if updated_result.modified_count == 0:
        raise HTTPException(status_code=404, detail="Item not found")
    return {"id": item_id, **shot.model_dump()}


@app.delete("/shots/{shot_id}")
async def delete_item(shot_id: str):
    delete_result = await collection.delete_one({"shot_id": shot_id})
    if delete_result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Item not found")
    return {"message": "Item deleted successfully"}
