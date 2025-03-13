import os
import requests


class DBClient:
    def __init__(self):
        self.url = os.environ["DB_URL"]

    def get_annotations(self):
        response = requests.get(f"{self.url}/shots/")
        items = response.json()
        return items

    def set_annotations(self, annotations: dict):
        requests.post(f"{self.url}/shots/", json=annotations)
