import os
from pathlib import Path
from argparse import ArgumentParser
import numpy
import random
from setup import create_project, create_local_samples, get_token
import pandas as pd


def create_mock_data(base_path: Path, shot_ids: list):
    data = {}
    for shot_id in shot_ids:
        t_ramp_start = random.randint(20, 40)
        t_ramp_end = random.randint(140, 240)
        t_disrupt_start = random.randint(400, 500)

        spike_time = random.randint(1, 5)
        down_time = random.randint(10, 20)

        low = numpy.random.uniform(0, 2, t_ramp_start)

        ramping = random.uniform(0.1, 1) * numpy.arange(
            t_ramp_end - t_ramp_start
        ) + numpy.random.uniform(0, 2, t_ramp_end - t_ramp_start)

        flat = ramping[-1] + numpy.random.uniform(0, 5, t_disrupt_start - t_ramp_end)

        spike = (
            flat[-1]
            + 20 * numpy.arange(spike_time)
            + numpy.random.uniform(0, 2, spike_time)
        )

        down = numpy.linspace(spike[-1], 0, down_time) + numpy.random.uniform(
            0, 2, down_time
        )

        low_2 = numpy.random.uniform(
            0, 2, 600 - (t_disrupt_start + spike_time + down_time)
        )

        current = numpy.concatenate((low, ramping, flat, spike, down, low_2))
        time = 0.02 * numpy.arange(len(current))

        data[shot_id] = {
            "data": {
                "ip": {"values": current.tolist(), "times": time.tolist()},
            },
            "annotations": {
                "ramp_up_start": 0.02 * t_ramp_start,
                "ramp_up_end": 0.02 * t_ramp_end,
                "disruption": 0.02 * t_disrupt_start,
            },
        }
        df = pd.DataFrame(data={"ip": current}, index=time)
        df.to_parquet(base_path.joinpath(f"{shot_id}.parquet"))

    return data, time


def main():
    parser = ArgumentParser()
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

    num_samples = 200
    base_path = Path(__file__).parents[1].joinpath("data", "test", "mock_disruptions")
    base_path.mkdir(parents=True, exist_ok=True)
    data, time = create_mock_data(base_path, list(range(1, num_samples + 1)))

    project_id = create_project(
        "Mock Disruption Project",
        "time-series",
        "tabular",
        "uncertainty",
        token=token,
        base_url=args.url,
        time_max=time[-1],
    )
    # Make annotations to add at same time as sample
    annotations = {
        shot_id: [
            {
                "shot_id": shot_id,
                "validated": True,
                "label": "Disruption",
                "time": item["annotations"]["disruption"],
                "created_by": "manual",
            }
        ]
        for shot_id, item in data.items()
    }
    create_local_samples(
        project_id,
        list(range(1, num_samples + 1)),
        token=token,
        base_url=args.url,
        base_path="data/test/mock_disruptions",
        file_type="parquet",
        annotations=annotations,
        signals=["ip"],
    )
    # Create 100 non-annotated samples
    create_mock_data(base_path, list(range(num_samples + 1, num_samples + 101)))
    create_local_samples(
        project_id,
        list(range(num_samples + 1, num_samples + 100)),
        token=token,
        base_url=args.url,
        base_path="data/test/mock_disruptions",
        file_type="parquet",
        signals=["ip"],
    )


if __name__ == "__main__":
    main()
