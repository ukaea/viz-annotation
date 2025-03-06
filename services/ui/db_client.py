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
        
