import random
import os
import redis
from celery import Celery, Task
import pandas as pd
import redis
from db_client import DBClient
from models.elm_model.main import ELMModel


REDIS_HOST = os.environ['REDIS_HOST']

app = Celery('tasks',
    broker=f"redis://{REDIS_HOST}:6379/0",  # Redis as a message broker
    backend=f"redis://{REDIS_HOST}:6379/0"  # Redis as result backend 
)


redis_client = redis.Redis(host=f'{REDIS_HOST}', port=6379, db=0)
# Flush client at start to remove any stale messages.
redis_client.flushdb()

@app.task()
def run_elm_model():
    sources = pd.read_parquet('https://mastapp.site/parquet/level2/sources')
    sources = sources.loc[sources.name == "spectrometer_visible"]
    all_shots = sources.shot_id.values.tolist()

    random.shuffle(all_shots)
    for shot in all_shots:
        redis_client.lpush("shot_queue", shot)

    db = DBClient()
    model = ELMModel(all_shots)
    sample_counter = 0
    batch_size = 10

    while True:
        # Block until there has been batch_size annotation updates from the user
        print('Waiting for update...')
        while sample_counter < batch_size:
            redis_client.blpop('update_queue')
            sample_counter += 1
            print(f'Update {sample_counter}')

        sample_counter = 0

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
