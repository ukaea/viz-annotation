from typing import Generic, Type, TypeVar

T = TypeVar('T')

class Registry(Generic[T]):
    def __init__(self) -> None:
        self._plugins = {}

    def register(self, name: str, cls: Type[T]) -> T:
        self._plugins[name] = cls

    def create(self, name: str, *args, **kwargs) -> T:
        return self._plugins[name](*args, **kwargs)
