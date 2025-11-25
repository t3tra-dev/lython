from typing import Any, Callable, TypeVar

from . import prim

__all__ = ["prim", "native", "to_prim", "from_prim"]

T = TypeVar("T", bound=prim.prim)

def native(func: Callable[..., Any]) -> Callable[..., Any]: ...
def to_prim(value: object, prim_type: type[T]) -> T: ...
def from_prim(prim_value: prim.prim) -> object: ...
