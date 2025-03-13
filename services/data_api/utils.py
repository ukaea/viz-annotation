import os
import io
import redis
import torch


class RedisModelCache:
    def __init__(self):
        REDIS_HOST = os.environ["REDIS_HOST"]
        self.client = redis.Redis(host=REDIS_HOST, port=6379, db=0)

    def exists(self, name: str) -> bool:
        return self.client.exists(name)

    def save_state(self, name: str, state):
        buffer = io.BytesIO()

        torch.save(state, buffer)
        buffer.seek(0)

        self.client.set(name, buffer.getvalue())

    def load_state(self, name: str):
        model_data = self.client.get(name)
        buffer = io.BytesIO(model_data)
        return torch.load(buffer)
