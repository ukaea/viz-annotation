import pathlib

import pandas as pd
import pytest


@pytest.mark.asyncio
async def test_get_data(api_client, setup_db):
    response = await api_client.post(
        f"/projects/{setup_db['project_id_2']}/samples/{setup_db['sample_id_4']}/data"
    )
    # Should collect data from '10000.parquet' file
    # Should only collect Ip, not dalpha
    assert response.status_code == 200
    data = response.json()
    assert data["values"].get("Ip")
    assert not data["values"].get("dalpha")
    assert data["values"]["Ip"]["time"] == list(range(100))
    # Load data from parquet, check it matches
    df = pd.read_parquet(pathlib.Path(__file__).parents[2].joinpath("10000.parquet"))
    assert data["values"]["Ip"]["values"] == df.Ip.tolist()
