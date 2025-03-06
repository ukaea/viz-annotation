from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from celery.result import AsyncResult
from celery_app import add
# from runner import ModelRunner

app = FastAPI()
# runner = ModelRunner()

@app.post("/run/{name}")
async def run_model(name: str):
    task = add.delay(4, 4)
    return {'task_id': task.id}
    # runner.run(name)

@app.get("/query/{task_id}")
async def query(task_id: str):
    task_result = AsyncResult(task_id)
    if task_result.ready():
        return {"shot_id": task_result.result}
    else:
        return {'shot_id': -1}

