from typing import Callable, Literal, TypeVar

from . import prim

__all__ = ["prim", "native", "to_prim", "from_prim"]

T = TypeVar("T")
PrimT = TypeVar("PrimT", bound=prim.prim)
PrimF = TypeVar("PrimF", bound=Callable[..., prim.prim])

def native(
    *,
    gc: Literal["none", "shadow-stack", "rc"] = ...,
) -> Callable[[PrimF], PrimF]: ...
def to_prim(value: object, prim_type: type[PrimT]) -> PrimT: ...
def from_prim(prim_value: prim.prim) -> object: ...
