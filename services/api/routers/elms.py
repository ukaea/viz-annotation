from fastapi import APIRouter
from typing import List
from pathlib import Path

from api.crud.db import MongoDBClient
from api.crud.mast import S3DataReader
from api.schemas.elms import ELMShot
from api.core.annotators import AnnotatorType
from api.core.data_pool import DataPool
from api.core.model_runner import run_training, run_inference

DATA_PATH = Path('/data/elms')

router = APIRouter(prefix="/elms")
db = MongoDBClient()
data_pool = DataPool(DATA_PATH)
data_reader = S3DataReader(DATA_PATH)
    
@router.post("/annotations/{shot_id}")
async def create_item(
    shot_id: int, item: ELMShot, method: AnnotatorType = AnnotatorType.UNET
):
    print(f"Updating annotations for {shot_id}")
    # Insert or update annotation in database
    new_annotation: bool = await db.upsert(item)

    # Get validated shots and update data pool
    shot_ids = await db.get_validated_shot_ids()
    data_pool.set_validated(shot_ids)

    # Check if we have met the criteria for retraining
    if not (data_pool.retrain() and new_annotation):
        return

    # Retrain the model
    print("Retraining model")
    annotations = await db.get_validated_annotations()
    unlabelled_shots = data_pool.unlabelled_shots
    run_model(method, shot_ids, annotations, unlabelled_shots)


@app.get("/annotations", response_model=List[ELMShot])
async def get_items():
    return await db.list()


@app.get("/annotations/{shot_id}", response_model=ELMShot)
async def get_item(
    shot_id: str,
    method: AnnotatorType = AnnotatorType.UNET,
    prominence: float = 0.1,
    distance: int = 1,
    force: bool = False,
):
    annotation = await db.find(shot_id)

    if annotation is None or force:
        print(f"Using annotator {method}")
        params = {
            "prominence": prominence,
            "distance": distance,
            "height": 0.01,
        }
        future = run_inference.delay(method, shot_id, **params)
        peaks = future.get(timeout=45)
        regions = []  # We currently do not support regions
        annotation = ELMShot(shot_id=shot_id, elms=peaks, regions=regions)
    else:
        print("Using annotation from database")

    return annotation


@app.get("/models/train")
async def train_model(method: AnnotatorType = AnnotatorType.UNET):
    labelled_shot_ids = await db.get_validated_shot_ids()
    annotations = await db.get_validated_annotations()
    unlabelled_shot_ids = data_pool.unlabelled_shots
    data_pool.set_validated(labelled_shot_ids)
    return run_model(method, labelled_shot_ids, annotations, unlabelled_shot_ids)


def run_model(method, labelld_shot_ids, annotations, unlabelled_shot_ids):
    if len(labelld_shot_ids) == 0:
        return {"status": "No labelled data."}
    if not data_pool.currently_training:
        future = run_training.delay(
            method, labelld_shot_ids, annotations, unlabelled_shot_ids
        )
        data_pool.training_future = future
        return {"status": "Training model"}
    else:
        print(data_pool.training_future.state)
        return {"status": "Model already training"}


@app.get("/next")
def get_next_shot_id():
    print("Querying next shot")
    shot_id = data_pool.query()
    return {"shot_id": shot_id}


@app.get("/data/{shot_id}")
async def get_data(shot_id: int):
    data = data_reader.get_data(shot_id)

    payload = {
        "dalpha": data,
        "shot_id": shot_id,
    }

    return payload

@app.get("/data/disruption/{shot_id}")
async def get_disruption_data(shot_id: int):
    return {
        "ip": data_reader.get_disruption_data(shot_id),
        "shot_id": shot_id,
    }