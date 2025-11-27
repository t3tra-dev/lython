from typing import Any, Callable, Literal, TypeVar

from . import prim

__all__ = ["prim", "native", "to_prim", "from_prim"]

T = TypeVar("T")
PrimT = TypeVar("PrimT", bound=prim.prim)

def native(
    func: Callable[..., Any] | None = ...,
    /,
    *,
    gc: Literal["none", "shadow-stack", "rc"] = ...,
) -> Callable[[Callable[..., T]], Callable[..., T]]: ...
def to_prim(value: object, prim_type: type[PrimT]) -> PrimT: ...
def from_prim(prim_value: prim.prim) -> object: ...
