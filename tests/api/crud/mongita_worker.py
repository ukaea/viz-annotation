"""Helper invoked as a separate OS process by test_mongita_multiprocess.py.

Kept out of the test module so each worker starts from a clean interpreter, which is
what makes it a real test of cross-process behaviour: Mongita's DiskEngine caches are
per-process module state, so two threads or two engines in one interpreter would only
approximate the problem.

The worker warms its cache, then waits at a file barrier before writing. Without the
barrier, interpreter start-up (seconds) dwarfs the write loop (milliseconds) and the
workers never actually overlap -- which lets a broken store pass the test.

Usage: python mongita_worker.py <store_path> <prefix> <count> [--barrier]
"""

import asyncio
import sys
import time
from pathlib import Path

from toktagger.api.crud.mongita_client import AsyncMongitaClient

BARRIER_TIMEOUT_S = 120


def _wait_for_start(store_path: str, prefix: str) -> None:
    """Announce readiness, then block until the parent releases every worker at once."""
    Path(f"{store_path}.ready-{prefix}").write_text("1")
    start = Path(f"{store_path}.start")
    deadline = time.monotonic() + BARRIER_TIMEOUT_S
    while not start.exists():
        if time.monotonic() > deadline:
            raise TimeoutError("barrier was never released")
        time.sleep(0.01)


async def _insert(store_path: str, prefix: str, count: int, barrier: bool) -> None:
    client = AsyncMongitaClient(store_path)
    collection = client["annotate_db"]["projects"]
    try:
        # Warm this process's document index before the barrier, so that once writing
        # starts every worker holds a cache that the others are about to invalidate.
        await collection.count_documents({})

        if barrier:
            _wait_for_start(store_path, prefix)

        for i in range(count):
            await collection.insert_one({"name": f"{prefix}-{i}"})
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(
        _insert(
            sys.argv[1],
            sys.argv[2],
            int(sys.argv[3]),
            barrier="--barrier" in sys.argv[4:],
        )
    )
