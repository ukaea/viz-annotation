from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from runner import ModelRunner

app = FastAPI()
runner = ModelRunner()

@app.post("/run/{name}")
async def run_model(name: str):
    runner.run(name)

