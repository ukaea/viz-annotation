import os
import tempfile
from pathlib import Path

import pytest


@pytest.mark.asyncio
async def test_get_files(api_client, setup_db):
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create fake parquet files
        Path(os.path.join(temp_dir, "file1.parquet")).touch()
        Path(os.path.join(temp_dir, "file2.parquet")).touch()
        Path(os.path.join(temp_dir, "file3.txt")).touch()  # Non-parquet file

        response = await api_client.get(
            f"/paths/files?dir_path={temp_dir}&file_type=parquet"
        )

        assert response.status_code == 200
        file_names = response.json()
        assert len(file_names) == 2


@pytest.mark.asyncio
async def test_get_directories(api_client, setup_db):
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create subdirectories
        # Should not find the first dir, does not contain any files of specified type, but should find the other two
        Path(os.path.join(temp_dir, "dir1")).mkdir()
        Path(os.path.join(temp_dir, "dir2")).mkdir()
        Path(os.path.join(temp_dir, "dir2/file.txt")).touch()
        Path(os.path.join(temp_dir, "dir3")).mkdir()
        Path(os.path.join(temp_dir, "dir3/file.txt")).touch()
        # Create a file that should not be included
        Path(os.path.join(temp_dir, "file.txt")).touch()

        response = await api_client.get(
            f"/paths/directories?dir_path={temp_dir}&file_type=txt"
        )

        assert response.status_code == 200
        dir_names = response.json()
        assert len(dir_names) == 2
        # Verify directories are sorted
        assert dir_names == sorted(dir_names)
        # Verify actual directory paths
        for name in dir_names:
            assert Path(name).name in ["dir2", "dir3"]
