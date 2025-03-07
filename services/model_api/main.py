import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from celery.result import AsyncResult
from model_runner import run_elm_model, redis_client

app = FastAPI()


@app.post("/start")
async def start_model():
    task = run_elm_model.delay()
    return {'task_id': task.id}

@app.post("/update")
async def update_model():
    redis_client.lpush('update_queue', '_update_')

@app.get("/query/{task_id}")
async def query(task_id: str):
    item = redis_client.blpop('shot_queue')
    shot_id= item[1].decode()
    return {"shot_id": shot_id}

