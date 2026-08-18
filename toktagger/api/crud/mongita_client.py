import asyncio
import os
import re
from collections.abc import AsyncIterator, Callable, Iterable
from pathlib import Path
from typing import Any

from filelock import FileLock
from mongita import MongitaClientDisk


class AsyncMongitaClient:
    """Async, multi-process-safe wrapper around the embedded Mongita store.

    Mongita is a single-process database. `DiskEngine` loads a collection's document
    index (`$.file_attrs`) the first time it is asked for and then never re-reads it,
    and `put_metadata` writes that in-memory index straight back over the file. Two
    workers sharing a store therefore diverge immediately, and the divergence is not
    merely stale reads: whichever worker writes next replaces the other's index
    entries, leaving their documents in `$.data` but unreachable -- silent data loss.

    To make several gunicorn workers safe, this class keeps a version counter beside
    the store. Every operation runs inside the cross-process FileLock, and reads the
    counter first: if another worker has written since our last operation, all cached
    state is dropped and re-read from disk. Writes bump the counter. Read-only
    workloads leave the counter untouched, so caches stay warm.

    Note that the FileLock serialises every database operation across all workers, so
    extra workers add no database concurrency -- only parallelism for the work either
    side of it (data loading, response building). Use a real MongoDB if you need
    concurrent database throughput.
    """

    def __init__(
        self,
        db_path: str,
    ) -> None:
        path = Path(db_path)
        path.mkdir(parents=True, exist_ok=True)
        self._client = MongitaClientDisk(str(path))
        self._closed = False
        self._mutex = asyncio.Lock()
        self._file_lock = FileLock(str(path) + ".lock")
        # Kept beside the store rather than inside it, like the lock file, so that
        # dropping a database cannot delete the counter along with the data.
        self._version_path = Path(str(path) + ".version")
        # The store version our caches correspond to. None forces a refresh on the
        # first operation, which is free because nothing is cached yet.
        self._seen_version: int | None = None

    def __getitem__(self, name: str) -> "AsyncDatabase":
        return AsyncDatabase(self, name)

    def _read_version(self) -> int | None:
        """Return the store's current version, or None if it could not be read.

        None is treated as "somebody else may have written", which fails in the safe
        direction: caches are dropped and the store is re-read.
        """
        try:
            return int(self._version_path.read_text())
        except (OSError, ValueError):
            return None

    def _refresh_if_stale(self) -> None:
        """Drop cached state if another process has written since our last operation.

        `engine.close()` clears the document cache, collection metadata, the document
        index and the open file handles. The engine object itself is reused rather
        than replaced, so the Database and Collection objects that hold a reference
        to it stay valid, and the handles are reopened lazily on next use.
        """
        version = self._read_version()
        if version is None or version != self._seen_version:
            self._client.engine.close()
            self._seen_version = version

    def _bump_version(self) -> None:
        """Record that this worker has written, invalidating every other worker."""
        next_version = (self._read_version() or 0) + 1
        # Written via a temporary file so a crash mid-write cannot leave a truncated
        # counter. A missing or unreadable counter only costs an extra refresh.
        tmp_path = Path(str(self._version_path) + ".tmp")
        tmp_path.write_text(str(next_version))
        os.replace(tmp_path, self._version_path)
        self._seen_version = next_version

    async def _run(self, fn: Callable, *, mutates: bool = False) -> Any:
        """Run fn in a thread, holding the cross-process FileLock.

        asyncio.Lock prevents multiple coroutines in this worker from
        queuing up threads; FileLock prevents concurrent access from
        other gunicorn workers.

        The staleness check and the version bump happen under the same lock as fn, so
        another worker cannot interleave with a read-modify-write cycle. Callers that
        change the store must pass mutates=True, otherwise other workers will keep
        serving from caches this write has invalidated.
        """
        async with self._mutex:

            def _in_thread():
                with self._file_lock:
                    self._refresh_if_stale()
                    result = fn()
                    if mutates:
                        self._bump_version()
                    return result

            return await asyncio.to_thread(_in_thread)

    async def close(self) -> None:
        if self._closed:
            return
        await asyncio.to_thread(self._client.close)
        self._closed = True


class AsyncDatabase:
    def __init__(self, client: AsyncMongitaClient, name: str) -> None:
        self._client = client
        self._name = name
        self._sync_db = client._client[name]

    def __getitem__(self, name: str) -> "AsyncCollection":
        return AsyncCollection(self, name)

    @property
    def name(self) -> str:
        return self._name


class AsyncCollection:
    def __init__(self, database: AsyncDatabase, name: str) -> None:
        self._database = database
        self._name = name
        self._sync_col = database._sync_db[name]

    @property
    def name(self) -> str:
        return self._name

    async def insert_one(self, document: dict[str, Any]) -> Any:
        return await self._database._client._run(
            lambda: self._sync_col.insert_one(document), mutates=True
        )

    async def insert_many(self, documents: Iterable[dict[str, Any]]) -> Any:
        docs = list(documents)
        return await self._database._client._run(
            lambda: self._sync_col.insert_many(docs), mutates=True
        )

    async def find_one(
        self, filter: dict[str, Any] | None = None, *args, **kwargs
    ) -> dict[str, Any] | None:
        f = filter or {}
        return await self._database._client._run(
            lambda: self._sync_col.find_one(f, *args, **kwargs)
        )

    def find(
        self,
        filter: dict[str, Any] | None = None,
        *args,
        skip: int = 0,
        limit: int | None = None,
        sort: list[tuple[str, int]] | None = None,
        **kwargs,
    ) -> "AsyncCursor":
        async def _snapshot() -> list[dict[str, Any]]:
            mongo_filter = {}
            regex_filters = []

            # Separate regex conditions from normal filters
            if filter:
                for field, condition in filter.items():
                    if isinstance(condition, dict) and "$regex" in condition:
                        pattern = condition["$regex"]
                        flags = condition.get("$options", 0)
                        if flags == "i":
                            flags = re.IGNORECASE
                        regex_filters.append((field, re.compile(pattern, flags)))
                    else:
                        mongo_filter[field] = condition

            def _do_find():
                cursor = self._sync_col.find(mongo_filter or {}, *args, **kwargs)
                return list(cursor)

            results = await self._database._client._run(_do_find)

            # Apply regex filters
            if regex_filters:
                filtered = []
                for doc in results:
                    match = True
                    for field, regex in regex_filters:
                        value = str(doc.get(field, ""))
                        if not regex.search(value):
                            match = False
                            break
                    if match:
                        filtered.append(doc)
                results = filtered

            if sort:
                for key, direction in reversed(sort):
                    results.sort(key=lambda x: x.get(key), reverse=(direction < 0))
            if skip:
                results = results[skip:]
            if limit is not None and limit > 0:
                results = results[:limit]
            return results

        return AsyncCursor(_snapshot())

    async def update_one(
        self, filter: dict[str, Any], update: dict[str, Any], *args, **kwargs
    ) -> Any:
        return await self._database._client._run(
            lambda: self._sync_col.update_one(filter, update, *args, **kwargs),
            mutates=True,
        )

    async def update_many(
        self, filter: dict[str, Any], update: dict[str, Any], *args, **kwargs
    ) -> Any:
        return await self._database._client._run(
            lambda: self._sync_col.update_many(filter, update, *args, **kwargs),
            mutates=True,
        )

    async def delete_one(self, filter: dict[str, Any], *args, **kwargs) -> Any:
        return await self._database._client._run(
            lambda: self._sync_col.delete_one(filter, *args, **kwargs), mutates=True
        )

    async def delete_many(self, filter: dict[str, Any], *args, **kwargs) -> Any:
        return await self._database._client._run(
            lambda: self._sync_col.delete_many(filter, *args, **kwargs), mutates=True
        )

    async def count_documents(
        self, filter: dict[str, Any] | None = None, *args, **kwargs
    ) -> int:
        f = filter or {}
        return await self._database._client._run(
            lambda: self._sync_col.count_documents(f, *args, **kwargs)
        )

    async def create_index(self, keys: Any, *args, **kwargs) -> Any:
        return await self._database._client._run(
            lambda: self._sync_col.create_index(keys, *args, **kwargs), mutates=True
        )


class AsyncCursor:
    def __init__(self, loader_coro: asyncio.Future | asyncio.Task | Any):
        self._loader = loader_coro
        self._docs: list[dict[str, Any]] | None = None
        self._idx = 0

    def __aiter__(self) -> AsyncIterator[dict[str, Any]]:
        self._idx = 0
        return self

    async def __anext__(self) -> dict[str, Any]:
        if self._docs is None:
            self._docs = await self._loader
        if self._idx >= len(self._docs):
            raise StopAsyncIteration
        doc = self._docs[self._idx]
        self._idx += 1
        return doc

    async def to_list(self, length: int | None = None) -> list[dict[str, Any]]:
        if self._docs is None:
            self._docs = await self._loader
        if length is None:
            return list(self._docs)
        return list(self._docs[:length])
