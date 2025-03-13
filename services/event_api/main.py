from fastapi import FastAPI, HTTPException
from typing import List
from model import Shot, ShotInDB
from client import MongoDBClient


# Initialize FastAPI app
app = FastAPI()

# MongoDB Client
client = MongoDBClient()


@app.post("/shots", response_model=ShotInDB)
async def create_item(item: Shot):
    return await client.upsert(item)


@app.get("/shots", response_model=List[ShotInDB])
async def get_items():
    return await client.list()


@app.get("/shots/{shot_id}", response_model=ShotInDB)
async def get_item(shot_id: str):
    item = await client.find(shot_id)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    return item


@app.delete("/shots/{shot_id}")
async def delete_item(shot_id: str):
    try:
        return client.delete(shot_id)
    except RuntimeError:
        raise HTTPException(status_code=404, detail="Item not found")
