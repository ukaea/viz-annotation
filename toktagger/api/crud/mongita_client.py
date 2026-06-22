import asyncio
import os
import re
from typing import Any, AsyncIterator, Callable, Dict, Iterable, List, Optional, Tuple
from filelock import FileLock
from mongita import MongitaClientDisk


class AsyncMongitaClient:
    def __init__(
        self,
        db_path: str,
    ) -> None:
        os.makedirs(db_path, exist_ok=True)
        self._client = MongitaClientDisk(db_path)
        self._closed = False
        self._mutex = asyncio.Lock()
        self._file_lock = FileLock(db_path + ".lock")

    def __getitem__(self, name: str) -> "AsyncDatabase":
        return AsyncDatabase(self, name)

    async def _run(self, fn: Callable) -> Any:
        """Run fn in a thread, holding the cross-process FileLock.

        asyncio.Lock prevents multiple coroutines in this worker from
        queuing up threads; FileLock prevents concurrent access from
        other gunicorn workers.
        """
        async with self._mutex:
            file_lock = self._file_lock

            def _in_thread():
                with file_lock:
                    return fn()

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

    async def insert_one(self, document: Dict[str, Any]) -> Any:
        return await self._database._client._run(
            lambda: self._sync_col.insert_one(document)
        )

    async def insert_many(self, documents: Iterable[Dict[str, Any]]) -> Any:
        docs = list(documents)
        return await self._database._client._run(
            lambda: self._sync_col.insert_many(docs)
        )

    async def find_one(
        self, filter: Optional[Dict[str, Any]] = None, *args, **kwargs
    ) -> Optional[Dict[str, Any]]:
        f = filter or {}
        return await self._database._client._run(
            lambda: self._sync_col.find_one(f, *args, **kwargs)
        )

    def find(
        self,
        filter: Optional[Dict[str, Any]] = None,
        *args,
        skip: int = 0,
        limit: Optional[int] = None,
        sort: Optional[List[Tuple[str, int]]] = None,
        **kwargs,
    ) -> "AsyncCursor":
        async def _snapshot() -> List[Dict[str, Any]]:
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
        self, filter: Dict[str, Any], update: Dict[str, Any], *args, **kwargs
    ) -> Any:
        return await self._database._client._run(
            lambda: self._sync_col.update_one(filter, update, *args, **kwargs)
        )

    async def update_many(
        self, filter: Dict[str, Any], update: Dict[str, Any], *args, **kwargs
    ) -> Any:
        return await self._database._client._run(
            lambda: self._sync_col.update_many(filter, update, *args, **kwargs)
        )

    async def delete_one(self, filter: Dict[str, Any], *args, **kwargs) -> Any:
        return await self._database._client._run(
            lambda: self._sync_col.delete_one(filter, *args, **kwargs)
        )

    async def delete_many(self, filter: Dict[str, Any], *args, **kwargs) -> Any:
        return await self._database._client._run(
            lambda: self._sync_col.delete_many(filter, *args, **kwargs)
        )

    async def count_documents(
        self, filter: Optional[Dict[str, Any]] = None, *args, **kwargs
    ) -> int:
        f = filter or {}
        return await self._database._client._run(
            lambda: self._sync_col.count_documents(f, *args, **kwargs)
        )

    async def create_index(self, keys: Any, *args, **kwargs) -> Any:
        return await self._database._client._run(
            lambda: self._sync_col.create_index(keys, *args, **kwargs)
        )


class AsyncCursor:
    def __init__(self, loader_coro: asyncio.Future | asyncio.Task | Any):
        self._loader = loader_coro
        self._docs: Optional[List[Dict[str, Any]]] = None
        self._idx = 0

    def __aiter__(self) -> AsyncIterator[Dict[str, Any]]:
        self._idx = 0
        return self

    async def __anext__(self) -> Dict[str, Any]:
        if self._docs is None:
            self._docs = await self._loader
        if self._idx >= len(self._docs):
            raise StopAsyncIteration
        doc = self._docs[self._idx]
        self._idx += 1
        return doc

    async def to_list(self, length: Optional[int] = None) -> List[Dict[str, Any]]:
        if self._docs is None:
            self._docs = await self._loader
        if length is None:
            return list(self._docs)
        return list(self._docs[:length])
