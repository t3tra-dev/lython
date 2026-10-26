# Overloaded functions and methods of a module the tests import.
from typing import overload


@overload
def parse(text: str) -> int: ...
@overload
def parse(text: str, as_float: bool) -> float: ...
def parse(text: str, as_float: bool = False) -> int | float:
    if as_float:
        return float(text)
    return int(text)


class Store:
    def __init__(self) -> None:
        self.items: dict[str, int] = {"a": 1}

    @overload
    def get(self, key: str) -> int | None: ...
    @overload
    def get(self, key: str, default: int) -> int: ...
    def get(self, key: str, default: int | None = None) -> int | None:
        if key in self.items:
            return self.items[key]
        return default
