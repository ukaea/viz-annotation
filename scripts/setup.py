import os
from argparse import ArgumentParser
from pathlib import Path
from typing import Optional
import requests


BASE_URL = "http://localhost:8002"


def get_token(base_url: str, username: str, password: str) -> str:
    r = requests.post(
        f"{base_url}/auth/token",
        data={"username": username, "password": password},
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    r.raise_for_status()
    return r.json()["access_token"]


def _auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def create_project(
    name: str,
    task: str,
    data_loader: str,
    query_strategy: str,
    token: str,
    base_url: str = BASE_URL,
    time_min: float = -0.1,
    time_max: float = 0.8,
    min_time_step: float = 0.0001,
) -> str:
    project = {
        "name": name,
        "task": task,
        "query_strategy": query_strategy,
        "data_loader": data_loader,
        "time_min": time_min,
        "time_max": time_max,
        "min_time_step": min_time_step,
    }

    response = requests.post(
        f"{base_url}/projects",
        json=project,
        headers=_auth(token),
    )
    response.raise_for_status()
    project_id = response.json()["_id"]
    return project_id


def create_uda_samples(
    project_id: str, shot_ids: list[int], token: str, base_url: str = BASE_URL
):
    samples = []
    for shot_id in shot_ids:
        sample = {
            "project_id": project_id,
            "shot_id": shot_id,
            "data": {
                "signal_names": ["ip", "ANE_DENSITY", "/xsx/HCAM/L/7"],
                "protocol": "uda",
            },
        }
        samples.append(sample)

    r = requests.post(
        f"{base_url}/projects/{project_id}/samples",
        json=samples,
        headers=_auth(token),
    )
    r.raise_for_status()


def create_sal_samples(
    project_id: str, shot_ids: list[int], token: str, base_url: str = BASE_URL
):
    samples = []
    for shot_id in shot_ids:
        sample = {
            "project_id": project_id,
            "shot_id": shot_id,
            "data": {
                "signal_names": ["ppf/signal/jetppf/magn/ipla"],
                "protocol": "sal",
            },
        }
        samples.append(sample)

    r = requests.post(
        f"{base_url}/projects/{project_id}/samples",
        json=samples,
        headers=_auth(token),
    )
    r.raise_for_status()


def create_fair_mast_samples(
    project_id: str, shot_ids: list[int], token: str, base_url: str = BASE_URL
):
    samples = []
    for shot_id in shot_ids:
        sample = {
            "project_id": project_id,
            "shot_id": shot_id,
            "data": {
                "signal_names": ["magnetics/ip"],
                "protocol": "fair_mast",
            },
        }
        samples.append(sample)

    r = requests.post(
        f"{base_url}/projects/{project_id}/samples",
        json=samples,
        headers=_auth(token),
    )
    r.raise_for_status()


def create_local_samples(
    project_id: str,
    shot_ids: list[int],
    token: str,
    base_url: str = BASE_URL,
    base_path: str = ".",
    file_type: str = "parquet",
    signals: Optional[list[str]] = None,
    annotations: Optional[list[dict]] = None,
):
    samples = []

    base_path = Path(base_path)
    for shot_id in shot_ids:
        file_name = str(base_path / f"{shot_id}.{file_type}")
        sample = {
            "shot_id": shot_id,
            "data": {
                "file_name": file_name,
                "type": file_type,
                "protocol": "file",
                "signal_names": signals,
            },
        }
        if annotations:
            sample["annotations"] = annotations[shot_id]
        samples.append(sample)

    r = requests.post(
        f"{base_url}/projects/{project_id}/samples",
        json=samples,
        headers=_auth(token),
    )
    r.raise_for_status()


def create_image_samples(
    project_id: str,
    shot_ids: list[int],
    image_dir: str,
    token: str,
    base_url: str = BASE_URL,
):
    samples = []
    for shot_id in shot_ids:
        samples.append(
            {
                "project_id": project_id,
                "shot_id": int(shot_id),
                "data": {
                    "file_name": str(
                        Path(image_dir) / str(shot_id)
                    ),  # directory, not a file
                    "type": "png",  # extension
                },
            }
        )

    r = requests.post(
        f"{base_url}/projects/{project_id}/samples",
        json=samples,
        headers=_auth(token),
    )
    r.raise_for_status()


def create_uda_camera_samples(
    project_id: str, shot_ids: list[int], token: str, base_url: str = BASE_URL
):
    samples = []
    for shot_id in shot_ids:
        sample = {
            "project_id": project_id,
            "shot_id": shot_id,
            "data": {
                "signal_names": ["rbb"],
                "protocol": "uda_camera",
            },
        }
        samples.append(sample)

    r = requests.post(
        f"{base_url}/projects/{project_id}/samples",
        json=samples,
        headers=_auth(token),
    )
    r.raise_for_status()


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "-b",
        "--base-path",
        default="./data/test",
        type=str,
        help="Base path for remote data files",
    )
    parser.add_argument(
        "--url",
        default=os.environ.get("TOKTAGGER_URL", "http://localhost:8002"),
        help="Base URL of the TokTagger API",
    )
    parser.add_argument(
        "--username",
        default=os.environ.get("TOKTAGGER_USERNAME", "admin"),
        help="Username for authentication",
    )
    parser.add_argument(
        "--password",
        default=os.environ.get("TOKTAGGER_PASSWORD"),
        required=not os.environ.get("TOKTAGGER_PASSWORD"),
        help="Password for authentication (or set TOKTAGGER_PASSWORD env var)",
    )
    args = parser.parse_args()

    token = get_token(args.url, args.username, args.password)

    base_path = Path(args.base_path)

    shot_files = Path("./data/test/summary").glob("*.parquet")
    shot_files = list(shot_files)
    shot_ids = [int(path.stem) for path in shot_files]

    project_id = create_project(
        "UDA Disruption Project",
        "time-series",
        "uda",
        "sequential",
        token=token,
        base_url=args.url,
    )
    create_uda_samples(project_id, shot_ids, token=token, base_url=args.url)

    project_id = create_project(
        "Local ELM Project",
        "time-series",
        "tabular",
        "sequential",
        token=token,
        base_url=args.url,
    )
    create_local_samples(
        project_id,
        shot_ids,
        token=token,
        base_url=args.url,
        base_path=base_path / "summary",
        file_type="parquet",
    )

    shot_files = Path("./data/test/mhd").glob("*.parquet")
    shot_files = list(shot_files)
    shot_ids = [int(path.stem) for path in shot_files]
    project_id = create_project(
        "Local MHD Project",
        "spectrogram",
        "tabular",
        "random",
        token=token,
        base_url=args.url,
        min_time_step=0.000001,
    )
    create_local_samples(
        project_id,
        shot_ids,
        token=token,
        base_url=args.url,
        base_path=base_path / "mhd",
        file_type="parquet",
        signals=["mirnov"],
    )
    # ---- Image / UFO demo project ----
    project_id = create_project(
        "Frame Project", "video", "image", "random", token=token, base_url=args.url
    )
    create_image_samples(
        project_id, [10101], Path("./data/test/video/"), token=token, base_url=args.url
    )

    # JET data
    project_id = create_project(
        "SAL Disruption Project",
        "time-series",
        "sal",
        query_strategy="sequential",
        token=token,
        base_url=args.url,
        time_min=38,
        time_max=None,
        min_time_step=0.0001,
    )
    shot_ids = [87737]
    create_sal_samples(project_id, shot_ids, token=token, base_url=args.url)

    # FAIR MAST
    project_id = create_project(
        "FAIR MAST Project",
        "time-series",
        "fair_mast",
        query_strategy="sequential",
        token=token,
        base_url=args.url,
    )
    shot_ids = [30421]
    create_fair_mast_samples(project_id, shot_ids, token=token, base_url=args.url)

    shot_ids = [30421]
    project_id = create_project(
        "UDA Camera Frame Project",
        "video",
        "uda_camera",
        "random",
        token=token,
        base_url=args.url,
    )
    create_uda_camera_samples(project_id, shot_ids, token=token, base_url=args.url)

    print("Projects and samples created successfully.")


if __name__ == "__main__":
    main()
