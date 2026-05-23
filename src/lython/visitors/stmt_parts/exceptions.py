from __future__ import annotations

import ast
from typing import cast

from ...mlir import ir
from ...mlir.dialects import _async_ops_gen as async_ops
from ...mlir.dialects import _lython_ops_gen as py_ops
from ...mlir.dialects import arith as arith_ops
from ...mlir.dialects import cf as cf_ops
from ...mlir.dialects import func as func_ops
from ..models import RegionBlocks
from .exception_analysis import (
    always_returns,
    contains_return,
    find_unsupported_finally_control,
    stmt_always_returns,
)
from .finally_returns import FinallyReturnMixin


class StmtExceptionMixin(FinallyReturnMixin):
    """Statement lowering for return and exception-related AST nodes."""

    def _prepare_return_value(self, node: ast.Return, loc: ir.Location) -> ir.Value:
        if node.value is None:
            if self._in_native_func:
                raise ValueError("@native function cannot return None implicitly")
            with loc, self.insertion_point():
                value = py_ops.NoneOp(self.get_py_type("!py.none")).result
        else:
            value = self.require_value(node.value, self.visit(node.value))

        with loc, self.insertion_point():
            expected_return = self.current_return_type()
            if expected_return is not None:
                value = self.coerce_value_to_type(value, expected_return, loc)
                if node.value is not None and str(expected_return).startswith(
                    "!py.func<"
                ):
                    returned_function_info = self.resolve_function_info_from_expression(
                        node.value, value
                    )
                    returned_callable_arg_index = (
                        self.resolve_current_function_parameter_index_from_expression(
                            node.value
                        )
                    )
                    if returned_callable_arg_index is not None:
                        returned_function_info = None
                    self._record_returned_callable_summary(
                        returned_function_info,
                        returned_callable_arg_index,
                    )
        return value

    def _emit_return_op(self, value: ir.Value, loc: ir.Location) -> None:
        with loc, self.insertion_point():
            if self._in_native_func:
                func_ops.ReturnOp([value])
            elif self.in_async_function():
                async_ops.ReturnOp([value])
            else:
                py_ops.ReturnOp([value])
        self._advance_block_after_terminator()

    def visit_Return(self, node: ast.Return) -> None:
        loc = self._loc(node)
        value = self._prepare_return_value(node, loc)
        if self._current_finally_return_context() is not None:
            self._emit_finally_return_yield(value, loc)
            return
        self._emit_return_op(value, loc)

    def visit_Raise(self, node: ast.Raise) -> None:
        if self._in_native_func:
            raise NotImplementedError("raise in @native is not supported")

        with self._loc(node), self.insertion_point():
            finally_context = self._current_finally_return_context()
            if finally_context is not None and finally_context.get("swallow_raise"):
                signal_type = finally_context["signal_type"]
                yield_kind = finally_context["yield_kind"]
                return_type = finally_context["return_type"]
                if return_type is None:
                    raise RuntimeError("invalid finally raise context")
                self._emit_finally_fallthrough_yield(
                    yield_kind, return_type, signal_type, self._loc(node)
                )
                return
            self._note_maythrow()
            if node.exc is None:
                py_ops.RaiseCurrentOp()
                self._advance_block_after_terminator()
                return

            if node.cause is not None:
                raise NotImplementedError("raise ... from ... is not supported")

            if isinstance(node.exc, ast.Constant) and isinstance(node.exc.value, str):
                exc_value = self.build_exception_value(
                    message=node.exc.value,
                    loc=self._loc(node),
                    context=self.current_exception_context(),
                )
                py_ops.RaiseOp(exc_value)
                self._advance_block_after_terminator()
                return

            if isinstance(node.exc, ast.Call):
                if (
                    isinstance(node.exc.func, ast.Name)
                    and node.exc.func.id == "Exception"
                ):
                    if len(node.exc.args) != 1 or not isinstance(
                        node.exc.args[0], ast.Constant
                    ):
                        raise NotImplementedError(
                            "Exception(...) requires a single string literal"
                        )
                    if not isinstance(node.exc.args[0].value, str):
                        raise NotImplementedError(
                            "Exception(...) requires a string literal"
                        )
                    exc_value = self.build_exception_value(
                        message=node.exc.args[0].value,
                        loc=self._loc(node),
                        context=self.current_exception_context(),
                    )
                    py_ops.RaiseOp(exc_value)
                    self._advance_block_after_terminator()
                    return

            raise NotImplementedError(
                "raise requires a string literal or bare raise (for now)"
            )

    def visit_Try(self, node: ast.Try) -> None:
        if node.handlers:
            if node.orelse and node.finalbody:
                unsupported = find_unsupported_finally_control(
                    [*node.body, *node.orelse, *node.finalbody]
                )
                for handler in node.handlers:
                    unsupported = unsupported or find_unsupported_finally_control(
                        handler.body
                    )
                if unsupported is not None:
                    raise NotImplementedError(
                        f"{unsupported} inside try/except/else/finally is not supported yet"
                    )

                loc = self._loc(node)
                with loc, self.insertion_point():
                    outer_try = py_ops.TryOp([])

                try_block = outer_try.try_region.blocks.append()
                finally_block = outer_try.finally_region.blocks.append()
                prev_block = self.current_block

                self._set_insertion_block(try_block)
                self.push_scope()
                inner_try = ast.Try(
                    body=node.body,
                    handlers=node.handlers,
                    orelse=node.orelse,
                    finalbody=[],
                )
                ast.copy_location(inner_try, node)
                self.visit_Try(inner_try)
                try_end_block = self.current_block
                try_terminated = try_end_block is None or self._block_terminated(
                    try_end_block
                )
                self.pop_scope()
                if try_end_block is not None and not try_terminated:
                    self._set_insertion_block(try_end_block)
                    with loc, self.insertion_point():
                        py_ops.TryYieldOp([])

                self._set_insertion_block(finally_block)
                self.push_scope()
                for stmt in node.finalbody:
                    self.visit(stmt)
                finally_end_block = self.current_block
                finally_terminated = (
                    finally_end_block is None
                    or self._block_terminated(finally_end_block)
                )
                self.pop_scope()
                if finally_end_block is not None and not finally_terminated:
                    self._set_insertion_block(finally_end_block)
                    with loc, self.insertion_point():
                        py_ops.FinallyYieldOp()

                self._set_insertion_block(prev_block)
                return

            if len(node.handlers) != 1:
                raise NotImplementedError(
                    "only a single except handler is supported yet"
                )
            handler = node.handlers[0]
            if handler.type is not None:
                if not (
                    isinstance(handler.type, ast.Name)
                    and handler.type.id == "Exception"
                ):
                    raise NotImplementedError(
                        "only bare except or except Exception is supported yet"
                    )
            finalbody_always_returns = bool(node.finalbody) and always_returns(
                node.finalbody
            )
            if node.finalbody:
                unsupported = (
                    find_unsupported_finally_control(
                        node.body, allow_raise=True, allow_return=True
                    )
                    or find_unsupported_finally_control(
                        handler.body, allow_raise=True, allow_return=True
                    )
                    or find_unsupported_finally_control(
                        node.finalbody, allow_return=finalbody_always_returns
                    )
                )
                if unsupported is not None:
                    raise NotImplementedError(
                        f"{unsupported} inside try/except/finally is not supported yet"
                    )

            has_else = bool(node.orelse)
            loc = self._loc(node)
            i1 = ir.IntegerType.get_signless(1, context=self.ctx)
            return_type = self.current_return_type()
            returns_via_try = (
                not has_else
                and return_type is not None
                and (
                    contains_return([*node.body, *handler.body])
                    or finalbody_always_returns
                )
            )
            returns_type = return_type if returns_via_try else None
            result_types = (
                [i1, returns_type]
                if returns_type is not None
                else ([i1] if has_else else [])
            )
            with loc, self.insertion_point():
                try_op = py_ops.TryOp(result_types)

            with loc:
                try_block = try_op.try_region.blocks.append()
                except_block = try_op.except_region.blocks.append(
                    self.get_py_type("!py.exception")
                )
                if node.finalbody:
                    if returns_type is not None:
                        finally_block = try_op.finally_region.blocks.append(
                            *result_types
                        )
                    else:
                        finally_block = try_op.finally_region.blocks.append()
                else:
                    finally_block = None
            prev_block = self.current_block

            self._set_insertion_block(try_block)
            self.push_scope()
            if returns_type is not None:
                self._push_finally_return_context("try", i1)
            for stmt in node.body:
                self.visit(stmt)
            if returns_type is not None:
                self._pop_finally_return_context()
            try_end_block = self.current_block
            try_terminated = try_end_block is None or self._block_terminated(
                try_end_block
            )
            self.pop_scope()
            if try_end_block is not None and not try_terminated:
                self._set_insertion_block(try_end_block)
                with loc, self.insertion_point():
                    if has_else:
                        completed = arith_ops.ConstantOp(
                            i1, ir.IntegerAttr.get(i1, 1)
                        ).result
                        py_ops.TryYieldOp([completed])
                    elif returns_type is not None:
                        self._emit_finally_fallthrough_yield(
                            "try", returns_type, i1, loc
                        )
                    else:
                        py_ops.TryYieldOp([])

            self._set_insertion_block(except_block)
            self.push_scope()
            if handler.name is not None:
                with loc, self.insertion_point():
                    exception_object = py_ops.UpcastOp(
                        self.get_py_type("!py.object"), except_block.arguments[0]
                    ).result
                self.define_symbol(handler.name, exception_object)
            self.push_exception_context(except_block.arguments[0])
            if returns_type is not None:
                self._push_finally_return_context(
                    "except",
                    i1,
                    return_type=returns_type,
                    swallow_raise=finalbody_always_returns,
                )
            try:
                for stmt in handler.body:
                    self.visit(stmt)
            finally:
                if returns_type is not None:
                    self._pop_finally_return_context()
                self.pop_exception_context()
            except_end_block = self.current_block
            except_terminated = except_end_block is None or self._block_terminated(
                except_end_block
            )
            self.pop_scope()
            if except_end_block is not None and not except_terminated:
                self._set_insertion_block(except_end_block)
                with loc, self.insertion_point():
                    if has_else:
                        completed = arith_ops.ConstantOp(
                            i1, ir.IntegerAttr.get(i1, 0)
                        ).result
                        py_ops.ExceptYieldOp([completed])
                    elif returns_type is not None:
                        self._emit_finally_fallthrough_yield(
                            "except", returns_type, i1, loc
                        )
                    else:
                        py_ops.ExceptYieldOp([])

            if finally_block is not None:
                self._set_insertion_block(finally_block)
                self.push_scope()
                if returns_type is not None and finalbody_always_returns:
                    self._push_finally_return_context("finally", i1)
                for stmt in node.finalbody:
                    self.visit(stmt)
                if returns_type is not None and finalbody_always_returns:
                    self._pop_finally_return_context()
                finally_end_block = self.current_block
                finally_terminated = (
                    finally_end_block is None
                    or self._block_terminated(finally_end_block)
                )
                self.pop_scope()
                if finally_end_block is not None and not finally_terminated:
                    self._set_insertion_block(finally_end_block)
                    with loc, self.insertion_point():
                        operands = (
                            list(finally_block.arguments)
                            if returns_type is not None
                            else []
                        )
                        py_ops.FinallyYieldOp(operands)

            self._set_insertion_block(prev_block)
            if returns_type is not None:
                self._emit_finally_return_dispatch(
                    try_op,
                    returns_type,
                    loc,
                    always_returns=finalbody_always_returns
                    or stmt_always_returns(node),
                )
                return
            if has_else:
                assert self.current_block is not None
                parent_region = self.current_block.region
                blocks = cast(RegionBlocks, parent_region.blocks)
                else_block = blocks.append()
                merge_block = blocks.append()
                with loc, self.insertion_point():
                    cf_ops.CondBranchOp(
                        try_op.results[0], [], [], else_block, merge_block
                    )

                self._set_insertion_block(else_block)
                self.push_scope()
                for stmt in node.orelse:
                    self.visit(stmt)
                else_terminated = self._block_terminated(else_block)
                self.pop_scope()
                if not else_terminated:
                    with loc, ir.InsertionPoint(else_block):
                        cf_ops.BranchOp([], merge_block)
                    self._set_insertion_block(merge_block)
                else:
                    self._set_insertion_block(merge_block)
            return
        if node.orelse:
            raise NotImplementedError("try/else requires except handlers")
        if not node.finalbody:
            raise NotImplementedError("try without finally is not supported yet")

        finalbody_always_returns = always_returns(node.finalbody)
        unsupported = find_unsupported_finally_control(
            node.body, allow_raise=True, allow_return=True
        ) or find_unsupported_finally_control(
            node.finalbody, allow_return=finalbody_always_returns
        )
        if unsupported is not None:
            raise NotImplementedError(
                f"{unsupported} inside try/finally is not supported yet"
            )

        loc = self._loc(node)
        i1 = ir.IntegerType.get_signless(1, context=self.ctx)
        return_type = self.current_return_type()
        finally_return = return_type is not None and (
            contains_return(node.body) or finalbody_always_returns
        )
        finally_return_type = return_type if finally_return else None
        result_types = (
            [i1, finally_return_type] if finally_return_type is not None else []
        )
        with loc, self.insertion_point():
            try_op = py_ops.TryOp(result_types)

        with loc:
            try_block = try_op.try_region.blocks.append()
            if finally_return_type is not None:
                finally_block = try_op.finally_region.blocks.append(*result_types)
            else:
                finally_block = try_op.finally_region.blocks.append()
        prev_block = self.current_block

        self._set_insertion_block(try_block)
        self.push_scope()
        if finally_return_type is not None:
            self._push_finally_return_context(
                "try",
                i1,
                return_type=finally_return_type,
                swallow_raise=finalbody_always_returns,
            )
        for stmt in node.body:
            self.visit(stmt)
        if finally_return_type is not None:
            self._pop_finally_return_context()
        try_end_block = self.current_block
        try_terminated = try_end_block is None or self._block_terminated(try_end_block)
        self.pop_scope()
        if try_end_block is not None and not try_terminated:
            self._set_insertion_block(try_end_block)
            with loc, self.insertion_point():
                if finally_return_type is not None:
                    self._emit_finally_fallthrough_yield(
                        "try", finally_return_type, i1, loc
                    )
                else:
                    py_ops.TryYieldOp([])

        self._set_insertion_block(finally_block)
        self.push_scope()
        if finally_return_type is not None and finalbody_always_returns:
            self._push_finally_return_context("finally", i1)
        for stmt in node.finalbody:
            self.visit(stmt)
        if finally_return_type is not None and finalbody_always_returns:
            self._pop_finally_return_context()
        finally_end_block = self.current_block
        finally_terminated = finally_end_block is None or self._block_terminated(
            finally_end_block
        )
        self.pop_scope()
        if finally_end_block is not None and not finally_terminated:
            self._set_insertion_block(finally_end_block)
            with loc, self.insertion_point():
                operands = (
                    list(finally_block.arguments)
                    if finally_return_type is not None
                    else []
                )
                py_ops.FinallyYieldOp(operands)

        self._set_insertion_block(prev_block)
        if finally_return_type is not None:
            self._emit_finally_return_dispatch(
                try_op,
                finally_return_type,
                loc,
                always_returns=finalbody_always_returns,
            )

    def visit_TryStar(self, node: ast.TryStar) -> None:
        raise NotImplementedError("Try star statement not implemented")

    def visit_Assert(self, node: ast.Assert) -> None:
        raise NotImplementedError("Assert statement not implemented")
