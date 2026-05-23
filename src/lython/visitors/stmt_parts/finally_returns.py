from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, cast

from ...mlir import ir
from ...mlir.dialects import _lython_ops_gen as py_ops
from ...mlir.dialects import arith as arith_ops
from ...mlir.dialects import cf as cf_ops
from ..models import FinallyReturnContext, RegionBlocks

if TYPE_CHECKING:
    from ..contracts import VisitorRuntime
else:
    VisitorRuntime = object


class _TryOpWithResults(Protocol):
    @property
    def results_(self) -> list[ir.Value]: ...


class FinallyReturnMixin(VisitorRuntime):
    """Return-value propagation helpers for try/finally lowering."""

    def _current_finally_return_context(self) -> FinallyReturnContext | None:
        if not self._finally_return_stack:
            return None
        return self._finally_return_stack[-1]

    def _return_signal_constant(
        self, signal_type: ir.Type, value: int, loc: ir.Location
    ) -> ir.Value:
        with loc, self.insertion_point():
            return arith_ops.ConstantOp(
                signal_type, ir.IntegerAttr.get(signal_type, value)
            ).result

    def _build_finally_return_seed(
        self, return_type: ir.Type, loc: ir.Location
    ) -> ir.Value:
        expr_visitor = self.subvisitors.get("Expr")
        if expr_visitor is not None:
            expr_visitor.current_block = self.current_block
            value = expr_visitor._build_invoke_result_seed(return_type, loc)
            self.current_block = expr_visitor.current_block
            return value

        type_str = str(return_type)
        with loc, self.insertion_point():
            if type_str == "!py.none":
                return py_ops.NoneOp(return_type).result
            if type_str in {"!py.int", "!py.bool"}:
                return py_ops.IntConstantOp(return_type, "0").result
            if type_str == "!py.float":
                return py_ops.FloatConstantOp(return_type, 0.0).result
            if type_str.startswith("i"):
                return arith_ops.ConstantOp(
                    return_type, ir.IntegerAttr.get(return_type, 0)
                ).result
            if type_str.startswith("f"):
                return arith_ops.ConstantOp(
                    return_type, ir.FloatAttr.get(return_type, 0.0)
                ).result
        raise NotImplementedError(
            f"finally return placeholder for {return_type} is not implemented yet"
        )

    def _emit_finally_return_yield(self, value: ir.Value, loc: ir.Location) -> None:
        context = self._current_finally_return_context()
        if context is None:
            raise RuntimeError("finally return context is not active")
        signal_type = context["signal_type"]
        yield_kind = context["yield_kind"]
        signal = self._return_signal_constant(signal_type, 1, loc)
        with loc, self.insertion_point():
            if yield_kind == "try":
                py_ops.TryYieldOp([signal, value])
            elif yield_kind == "except":
                py_ops.ExceptYieldOp([signal, value])
            elif yield_kind == "finally":
                py_ops.FinallyYieldOp([signal, value])
            else:
                raise RuntimeError(f"unknown finally return yield kind: {yield_kind}")
        self._advance_block_after_terminator()

    def _emit_finally_fallthrough_yield(
        self,
        yield_kind: str,
        return_type: ir.Type,
        signal_type: ir.Type,
        loc: ir.Location,
    ) -> None:
        signal = self._return_signal_constant(signal_type, 0, loc)
        seed = self._build_finally_return_seed(return_type, loc)
        with loc, self.insertion_point():
            if yield_kind == "try":
                py_ops.TryYieldOp([signal, seed])
            elif yield_kind == "except":
                py_ops.ExceptYieldOp([signal, seed])
            else:
                raise RuntimeError(
                    f"unsupported finally fallthrough yield: {yield_kind}"
                )

    def _emit_finally_return_dispatch(
        self,
        try_op: _TryOpWithResults,
        return_type: ir.Type,
        loc: ir.Location,
        *,
        always_returns: bool = False,
    ) -> None:
        results = try_op.results_
        if always_returns:
            self._emit_return_op(results[1], loc)
            return

        assert self.current_block is not None
        parent_region = self.current_block.region
        blocks = cast(RegionBlocks, parent_region.blocks)
        return_block = blocks.append()
        merge_block = blocks.append()
        with loc, self.insertion_point():
            cf_ops.CondBranchOp(results[0], [], [], return_block, merge_block)

        self._set_insertion_block(return_block)
        self._emit_return_op(results[1], loc)
        self._set_insertion_block(merge_block)

    def _push_finally_return_context(
        self,
        yield_kind: str,
        signal_type: ir.Type,
        *,
        return_type: ir.Type | None = None,
        swallow_raise: bool = False,
    ) -> None:
        self._finally_return_stack.append(
            {
                "yield_kind": yield_kind,
                "signal_type": signal_type,
                "return_type": return_type,
                "swallow_raise": swallow_raise,
            }
        )

    def _pop_finally_return_context(self) -> None:
        self._finally_return_stack.pop()
