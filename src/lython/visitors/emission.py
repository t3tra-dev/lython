from __future__ import annotations

import ast
from collections.abc import Sequence
from typing import TYPE_CHECKING, cast

from ..frontend.locations import source_position
from ..mlir import ir
from .models import ArrayAttrFactory, AttributeCarrier, FinallyReturnContext

if TYPE_CHECKING:
    from .contracts import VisitorRuntime
else:
    VisitorRuntime = object


class EmissionMixin(VisitorRuntime):
    """Shared MLIR emission state and small IR construction utilities."""

    def _set_module(self, module: ir.Module) -> None:
        self.module = module
        for visitor in self.subvisitors.values():
            visitor.module = module

    def _set_module_name(self, name: str) -> None:
        self._module_name = name
        for visitor in self.subvisitors.values():
            visitor._module_name = name

    def _set_insertion_block(self, block: ir.Block | None) -> None:
        self.current_block = block
        for visitor in self.subvisitors.values():
            visitor.current_block = block

    def _set_finally_return_stack(self, stack: list[FinallyReturnContext]) -> None:
        self._finally_return_stack = stack
        for visitor in self.subvisitors.values():
            visitor._finally_return_stack = stack

    def _set_exception_context_stack(self, stack: list[ir.Value]) -> None:
        self._exception_context_stack = stack
        for visitor in self.subvisitors.values():
            visitor._exception_context_stack = stack

    def _set_func_effect(self, func: AttributeCarrier, maythrow: bool) -> None:
        if "nothrow" in func.attributes:
            del func.attributes["nothrow"]
        if "maythrow" in func.attributes:
            del func.attributes["maythrow"]
        if maythrow:
            func.attributes["maythrow"] = ir.UnitAttr.get(self.ctx)
        else:
            func.attributes["nothrow"] = ir.UnitAttr.get(self.ctx)

    def get_py_type(self, type_spec: str) -> ir.Type:
        cached = self._type_cache.get(type_spec)
        if cached is None:
            cached = ir.Type.parse(type_spec, self.ctx)
            self._type_cache[type_spec] = cached
        return cached

    def require_value(self, node: ast.AST, result: object) -> ir.Value:
        if isinstance(result, ir.Value):
            if self._in_native_func and result in self._prim_deallocated:
                loc = self._loc(node)
                raise ValueError(
                    f"Use-after-dealloc detected in @native(gc={self._native_gc_mode}): "
                    f"{loc}"
                )
            return result
        raise TypeError(
            f"Visitor for {type(node).__name__} must return an MLIR value, got {type(result)!r}"
        )

    def _loc(self, node: ast.AST) -> ir.Location:
        position = source_position(node)
        if position is None:
            return ir.Location.unknown(self.ctx)
        lineno, col = position
        return ir.Location.file(self._module_name, int(lineno), int(col) + 1, self.ctx)

    def insertion_point(self) -> ir.InsertionPoint:
        if self.current_block is None:
            raise RuntimeError("Insertion block is not set")
        return ir.InsertionPoint(self.current_block)

    def array_attr(self, attributes: Sequence[ir.Attribute]) -> ir.ArrayAttr:
        array_attr_cls = cast(ArrayAttrFactory, ir.ArrayAttr)
        return array_attr_cls.get(list(attributes), context=self.ctx)

    def _block_terminated(self, block: ir.Block) -> bool:
        ops = list(block.operations)
        if not ops:
            return False
        terminators = {
            "py.return",
            "py.raise",
            "py.raise.current",
            "py.invoke",
            "py.try.yield",
            "py.except.yield",
            "py.finally.yield",
            "async.return",
            "func.return",
            "cf.br",
            "cf.cond_br",
        }
        return ops[-1].operation.name in terminators

    def _advance_block_after_terminator(self) -> None:
        """Move insertion to a fresh block if the current block is terminated."""
        block = self.current_block
        if block is None:
            return
        ops = list(block.operations)
        if not ops:
            return
        terminators = {
            "py.return",
            "py.raise",
            "py.raise.current",
            "py.invoke",
            "py.try.yield",
            "py.except.yield",
            "py.finally.yield",
            "async.return",
            "func.return",
            "cf.br",
            "cf.cond_br",
        }
        if ops[-1].operation.name not in terminators:
            return
        # Do not create an unreachable empty block. Callers should
        # explicitly set a new insertion block when control can continue.
        self._set_insertion_block(None)
