import os
import requests

class DBClient():
    def __init__(self):
        self.url = os.environ['DB_URL']

    def save(self, shot_id: int, elms: dict):
        payload = {
            'shot_id': shot_id
        }

        items = []
        for elm in elms:
            item = {'time': elm['Time'], 'valid': elm['Valid'], 'height': elm['Height']}
            items.append(item)
        payload['elms'] = items

        requests.post(f"{self.url}/shots", json=payload)

class ModelClient:
    def __init__(self):
        self.url = os.environ['MODEL_API_URL']

    def start_training(self):
        response = requests.post(f"{self.url}/start")
        response = response.json()
        task_id = response['task_id']
        return task_id

    def update(self):
        requests.post(f"{self.url}/update")
        
    def query(self, task_id: int) -> int:
        response = requests.get(f"{self.url}/query/{task_id}")
        response= response.json()
        return response['shot_id']
        
