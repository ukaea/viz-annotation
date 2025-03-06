from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel

app = FastAPI()

@app.post("/run/{name}")
async def run_model(name: str):
    print("Model Name:", name)

