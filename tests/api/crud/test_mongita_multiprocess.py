"""Cross-process safety of the embedded Mongita store.

Mongita is a single-process database: DiskEngine reads a collection's document index
($.file_attrs) once and never re-reads it, and put_metadata writes that in-memory
index back over the file. Running several Gunicorn workers against one store therefore
used to lose documents outright -- whichever worker wrote next replaced the other
workers' index entries, orphaning their documents inside $.data.

AsyncMongitaClient keeps a version counter beside the store and drops its caches when
another process has written. These tests drive that with real subprocesses, since the
caches are per-process state that threads would share.
"""

import subprocess
import sys
import time
from pathlib import Path

import pytest

from toktagger.api.crud.mongita_client import AsyncMongitaClient

WORKER = Path(__file__).parent / "mongita_worker.py"
REPO_ROOT = Path(__file__).resolve().parents[3]
BARRIER_TIMEOUT_S = 120


def _spawn(
    store: Path, prefix: str, count: int, barrier: bool = False
) -> subprocess.Popen:
    args = [sys.executable, str(WORKER), str(store), prefix, str(count)]
    if barrier:
        args.append("--barrier")
    return subprocess.Popen(
        args, cwd=REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )


def _release_barrier(store: Path, prefixes: list[str]) -> None:
    """Wait until every worker has warmed its cache, then start them all together."""
    deadline = time.monotonic() + BARRIER_TIMEOUT_S
    while time.monotonic() < deadline:
        if all(Path(f"{store}.ready-{p}").exists() for p in prefixes):
            break
        time.sleep(0.05)
    else:
        raise AssertionError("workers never reached the barrier")
    Path(f"{store}.start").write_text("go")


@pytest.mark.asyncio
async def test_concurrent_writers_do_not_lose_documents(tmp_path):
    """Every document written by every worker survives.

    All workers hold a warm document index and then write at once, so each write lands
    on a store another worker believes it already knows. Without cross-process
    invalidation, each worker's put_metadata writes its stale index over the others'
    and their documents become unreachable.
    """
    store = tmp_path / "toktagger_db"
    prefixes = ["w1", "w2", "w3"]
    per_worker = 15

    procs = [_spawn(store, prefix, per_worker, barrier=True) for prefix in prefixes]
    try:
        _release_barrier(store, prefixes)
    except AssertionError:
        for proc in procs:
            proc.kill()
        raise
    for proc in procs:
        _, stderr = proc.communicate(timeout=180)
        assert proc.returncode == 0, stderr.decode()

    client = AsyncMongitaClient(str(store))
    try:
        docs = await client["annotate_db"]["projects"].find({}).to_list()
    finally:
        await client.close()

    found = {doc["name"] for doc in docs}
    expected = {f"{p}-{i}" for p in prefixes for i in range(per_worker)}
    missing = expected - found
    assert not missing, f"{len(missing)} documents were lost: {sorted(missing)[:10]}"
    assert len(docs) == len(expected), (
        f"expected {len(expected)} docs, found {len(docs)}"
    )


@pytest.mark.asyncio
async def test_reader_sees_another_process_write(tmp_path):
    """A long-lived reader picks up a write made by another process.

    The read before the write is what makes this a real test: it warms this process's
    DiskEngine caches, and without invalidation the second read is served from them and
    never sees the new document.
    """
    store = tmp_path / "toktagger_db"
    client = AsyncMongitaClient(str(store))
    try:
        collection = client["annotate_db"]["projects"]
        await collection.insert_one({"name": "seed"})
        assert await collection.count_documents({}) == 1  # warms the caches

        proc = _spawn(store, "other", 1)
        _, stderr = proc.communicate(timeout=180)
        assert proc.returncode == 0, stderr.decode()

        names = {doc["name"] for doc in await collection.find({}).to_list()}
        assert names == {"seed", "other-0"}, f"stale read: {names}"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_reads_do_not_invalidate_a_warm_cache(tmp_path):
    """Read-only traffic leaves the version counter alone, so caches stay warm.

    Without this the fix would be correct but would re-read the whole store on every
    query, making extra workers slower than a single one.
    """
    store = tmp_path / "toktagger_db"
    client = AsyncMongitaClient(str(store))
    try:
        collection = client["annotate_db"]["projects"]
        await collection.insert_one({"name": "seed"})

        version_path = Path(str(store) + ".version")
        after_write = version_path.read_text()

        for _ in range(20):
            await collection.count_documents({})
            await collection.find({}).to_list()

        assert version_path.read_text() == after_write, (
            "reads bumped the version counter, which would flush every other worker"
        )
    finally:
        await client.close()
