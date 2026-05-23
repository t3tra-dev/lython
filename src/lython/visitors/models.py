from __future__ import annotations

from collections.abc import MutableMapping
from typing import Protocol, TypedDict

from ..frontend.program import TypedProgram
from ..frontend.symbols import ClassInfo, FunctionInfo, MethodInfo
from ..mlir import ir


class FinallyReturnContext(TypedDict):
    yield_kind: str
    signal_type: ir.Type
    return_type: ir.Type | None
    swallow_raise: bool


class NativeDecoratorInfo(TypedDict):
    gc: str


class AttributeCarrier(Protocol):
    attributes: MutableMapping[str, object]


class ArrayAttrFactory(Protocol):
    def get(
        self, attributes: list[ir.Attribute], *, context: ir.Context
    ) -> ir.ArrayAttr: ...


class BlockFactory(Protocol):
    def create_at_start(
        self, region: ir.Region, arg_types: list[ir.Type]
    ) -> ir.Block: ...


class RegionBlocks(Protocol):
    def append(self, *arg_types: ir.Type) -> ir.Block: ...


class AffineMapFactory(Protocol):
    def get(
        self, dim_count: int, symbol_count: int, exprs: list[ir.AffineExpr]
    ) -> ir.AffineMap: ...


class IteratorAttrBuilder(Protocol):
    def __call__(self, value: object, *, context: ir.Context) -> ir.Attribute: ...


class AttrBuilderFactory(Protocol):
    def get(self, kind: str) -> IteratorAttrBuilder: ...


VisitResult = ir.Value | None


__all__ = [
    "AffineMapFactory",
    "ArrayAttrFactory",
    "AttrBuilderFactory",
    "AttributeCarrier",
    "BlockFactory",
    "ClassInfo",
    "FinallyReturnContext",
    "FunctionInfo",
    "IteratorAttrBuilder",
    "MethodInfo",
    "NativeDecoratorInfo",
    "RegionBlocks",
    "TypedProgram",
    "VisitResult",
]
