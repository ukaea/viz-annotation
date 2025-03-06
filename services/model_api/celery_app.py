import time
import redis
import celery
from celery import Celery, Task
import numpy as np
import pandas as pd
import redis
from db_client import DBClient
from models.elm_model.main import ELMModel

app = Celery('tasks',
    broker="redis://redis:6379/0",  # Redis as a message broker
    backend="redis://redis:6379/0"  # Redis as result backend 
)


redis_client = redis.Redis(host='redis', port=6379, db=0)
redis_client.flushdb()

@app.task(bind=True)
def run_elm_model(self):
    sources = pd.read_parquet('https://mastapp.site/parquet/level2/sources')
    sources = sources.loc[sources.name == "spectrometer_visible"]
    all_shots = sources.shot_id.values.tolist()

    for shot in all_shots:
        redis_client.lpush("shot_queue", shot)

    db = DBClient()
    model = ELMModel(all_shots)

    while True:
        # Block until there is an update from the user
        print('Waiting for update...')
        redis_client.blpop('update_queue')

        # Get all annotated items for database
        print('Getting all annotations from DB')
        items = db.get_annotations()

        # Train model
        print('Train model')
        model.train(items)

        # Query model for next shot
        print('Querying model for next shot')
        next_shots = model.query(n_samples=10)

        for shot in next_shots:
            redis_client.lpush("shot_queue", int(shot))

        print(f'Next shot {shot}')
