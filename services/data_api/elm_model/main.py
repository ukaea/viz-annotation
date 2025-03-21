import json
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from elm_model.annotator import UnetELMDataAnnotator


def load_annotations(path: Path):
    path = Path(path)
    file_names = path.glob("*.json")

    annotations = {}
    for name in file_names:
        with name.open("r") as f:
            shot = name.stem
            data = json.load(f)
            events = data["elms"]["events"]
            for event in events:
                event["valid"] = True
            annotations[shot] = events
    return annotations


def main():
    epochs = 30
    seed = 42
    data_path = "../../data/elms"
    annotations_path = "../../data/elm-events"

    annotations = load_annotations(annotations_path)
    shots = list(annotations.keys())

    train_shots, test_shots = train_test_split(shots, random_state=seed)

    train_annotations = [{"elms": annotations[shot]} for shot in train_shots]
    test_annotations = [{"elms": annotations[shot]} for shot in test_shots]

    model = UnetELMDataAnnotator(epochs=epochs, data_path=data_path)
    model.train(train_shots, train_annotations)
    model.evaluate(test_shots, test_annotations)
    model.network.to("cpu")
    torch.save(model.network.state_dict(), "elm_model/model.pth")


if __name__ == "__main__":
    main()
