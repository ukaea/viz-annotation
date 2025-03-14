import os
import redis
from celery import Celery
from annotators import ANNOTATORS, DataAnnotator


REDIS_HOST = os.environ["REDIS_HOST"]

app = Celery(
    "tasks",
    broker=f"redis://{REDIS_HOST}:6379/0",  # Redis as a message broker
    backend=f"redis://{REDIS_HOST}:6379/0",  # Redis as result backend
)

redis_client = redis.Redis(host=f"{REDIS_HOST}", port=6379, db=0)
# Flush client at start to remove any stale messages.
redis_client.flushdb()


@app.task()
def run_annotator(method: str, labelled_shot_ids, annotations, unlabelled_shot_ids):
    print(f"Running annotator training {method}")
    annotator: DataAnnotator = ANNOTATORS[method]()
    annotator.train(labelled_shot_ids, annotations)
    scores = annotator.score(unlabelled_shot_ids)
    return scores


@app.task()
def run_inference(method: str, shot_id: int):
    print(f"Running annotator inference {method}")
    annotator: DataAnnotator = ANNOTATORS[method]()
    return annotator.get_annotations(shot_id)


@app.task()
def run_score(method: str, shot_ids: list[int]):
    print(f"Running annotator scoring {method}")
    annotator: DataAnnotator = ANNOTATORS[method]()
    return annotator.score(shot_ids)
