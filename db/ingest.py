from pathlib import Path
import requests
import json

URL = "http://localhost:8000/shots"


def load_json(file_name: Path):
    with file_name.open("r") as handle:
        return json.load(handle)


def main():
    data_path = Path("../notebooks/example-events")
    file_names = data_path.glob("*.json")

    for file_name in file_names:
        events = load_json(file_name)

        shot_id = int(file_name.stem)

        elms = [{"time": item["time"]} for item in events["elms"]["events"]]

        flat_top = {
            "time_min": events["general"]["flat_top"]["tmin"],
            "time_max": events["general"]["flat_top"]["tmax"],
        }

        ramp_up = {
            "time_min": events["general"]["ramp_up"]["tmin"],
            "time_max": events["general"]["ramp_up"]["tmax"],
        }

        item = {
            "shot_id": shot_id,
            "elms": elms,
            "flat_top": flat_top,
            "ramp_up": ramp_up,
        }

        print(item)
        requests.post(URL, json=item)


if __name__ == "__main__":
    main()
