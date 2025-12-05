from typing import Callable, Literal, TypeVar

from . import prim

__all__ = ["prim", "native", "to_prim", "from_prim"]

T = TypeVar("T")
PrimT = TypeVar("PrimT", bound=prim.Prim[prim.Int] | prim.Prim[prim.Float])
PrimFunc = TypeVar(
    "PrimFunc",
    bound=Callable[..., prim.Prim[prim.Int]] | Callable[..., prim.Prim[prim.Float]],
)

def native(
    *,
    gc: Literal["none", "shadow-stack", "rc"] = "none",
) -> Callable[[PrimFunc], PrimFunc]: ...
def to_prim(value: object, prim_type: type[PrimT]) -> PrimT: ...
def from_prim(
    prim_value: (
        prim.Prim[prim.Int]
        | prim.Prim[prim.Float]
        | prim.Vector
        | prim.Matrix
        | prim.Tensor
    ),
) -> object: ...
