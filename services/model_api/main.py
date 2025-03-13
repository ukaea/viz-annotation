from typing import Optional
from fastapi import FastAPI
from models.classic_model.tagger import ClassicELMModel
from db_client import DBClient

app = FastAPI()


@app.get("/next")
async def next_shot(shot_id: Optional[int] = None):
    if shot_id is None:
        # Query for next shot number
        next_shot_id = 30420
    else:
        next_shot_id = shot_id

    # Predict labels for next shot
    model = ClassicELMModel()
    annotations = model.predict(next_shot_id)

    # Update annotations in DB
    db = DBClient()
    db.set_annotations(annotations)

    # Return next shot number
    return {"shot_id": next_shot_id}
