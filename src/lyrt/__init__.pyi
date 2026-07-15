from typing import (
    Any,
    Awaitable,
    Callable,
    Generator,
    Literal,
    TypeVar,
    overload,
)

from . import prim

__all__ = [
    "prim",
    "native",
    "to_prim",
    "from_prim",
    "alloc",
    "dealloc",
    "ReadyIntAwaitable",
]

T = TypeVar("T")
AllocT = TypeVar(
    "AllocT",
    bound=prim.Int | prim.Float | prim.Vector | prim.Matrix | prim.Tensor,
)
PrimT = TypeVar("PrimT", bound=prim.Int | prim.Float)
PrimFunc = TypeVar(
    "PrimFunc",
    bound=Callable[..., prim.Int] | Callable[..., prim.Float],
)

type _NestedNumber = int | float | list[_NestedNumber]

class ReadyIntAwaitable(Awaitable[int]):
    def __init__(self, value: int) -> None: ...
    def __await__(self) -> Generator[Any, Any, int]: ...

def native(
    *,
    gc: Literal["none", "shadow-stack", "rc"] = "none",
) -> Callable[[PrimFunc], PrimFunc]: ...
def to_prim(value: object, prim_type: type[PrimT]) -> PrimT: ...
@overload
def from_prim(prim_value: prim.Int) -> int: ...
@overload
def from_prim(prim_value: prim.Float) -> float: ...
@overload
def from_prim(prim_value: prim.Vector) -> list[int | float]: ...
@overload
def from_prim(prim_value: prim.Matrix) -> list[list[int | float]]: ...
@overload
def from_prim(prim_value: prim.Tensor) -> list[_NestedNumber]: ...
def alloc(value: AllocT) -> AllocT: ...
def dealloc(
    value: prim.Int | prim.Float | prim.Vector | prim.Matrix | prim.Tensor,
) -> None: ...
