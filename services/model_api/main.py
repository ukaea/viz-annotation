from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI()

class ModelParams(BaseModel):
    name: str
    

@app.post("/run/")
async def run_model(params: ModelParams):
    print(params)

