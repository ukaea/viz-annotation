import os
import redis
from celery import Celery
from annotators import DataAnnotator, AnnotatorType
from elm_model.annotator import UnetELMDataAnnotator, ClassicELMDataAnnotator

ANNOTATORS = {
    AnnotatorType.CLASSIC: ClassicELMDataAnnotator,
    AnnotatorType.UNET: UnetELMDataAnnotator,
}


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
def run_training(method: str, labelled_shot_ids, annotations, unlabelled_shot_ids):
    print(f"Running annotator training {method}")
    annotator: DataAnnotator = ANNOTATORS[method]()
    annotator.load_model()
    annotator.train(labelled_shot_ids, annotations)
    annotator.save_model()
    scores = annotator.score(unlabelled_shot_ids)
    return scores


@app.task()
def run_inference(method: str, shot_id: int, **params):
    print(f"Running annotator inference {method}")
    annotator: DataAnnotator = ANNOTATORS[method]()
    annotator.load_model()
    return annotator.get_annotations(shot_id, **params)
