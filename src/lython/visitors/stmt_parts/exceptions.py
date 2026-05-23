# pyright: reportAttributeAccessIssue=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false
from __future__ import annotations

import ast

from ...mlir import ir
from ...mlir.dialects import arith as arith_ops
from ...mlir.dialects import _lython_ops_gen as py_ops
from ...mlir.dialects import _async_ops_gen as async_ops
from ...mlir.dialects import cf as cf_ops
from ...mlir.dialects import func as func_ops


class _UnsupportedFinallyControlDetector(ast.NodeVisitor):
    def __init__(
        self, *, allow_raise: bool = False, allow_return: bool = False
    ) -> None:
        self.allow_raise = allow_raise
        self.allow_return = allow_return
        self.unsupported: str | None = None

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        return

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        return

    def visit_Lambda(self, node: ast.Lambda) -> None:
        return

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        return

    def visit_Return(self, node: ast.Return) -> None:
        if self.allow_return:
            return
        self.unsupported = "return"

    def visit_Raise(self, node: ast.Raise) -> None:
        if self.allow_raise:
            return
        self.unsupported = "raise"

    def visit_Break(self, node: ast.Break) -> None:
        self.unsupported = "break"

    def visit_Continue(self, node: ast.Continue) -> None:
        self.unsupported = "continue"


def _find_unsupported_finally_control(
    nodes: list[ast.stmt], *, allow_raise: bool = False, allow_return: bool = False
) -> str | None:
    detector = _UnsupportedFinallyControlDetector(
        allow_raise=allow_raise, allow_return=allow_return
    )
    for node in nodes:
        detector.visit(node)
        if detector.unsupported is not None:
            return detector.unsupported
    return None


class _FinallyReturnDetector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.found = False

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        return

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        return

    def visit_Lambda(self, node: ast.Lambda) -> None:
        return

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        return

    def visit_Return(self, node: ast.Return) -> None:
        self.found = True


def _contains_return(nodes: list[ast.stmt]) -> bool:
    detector = _FinallyReturnDetector()
    for node in nodes:
        detector.visit(node)
        if detector.found:
            return True
    return False


def _stmt_always_returns(node: ast.stmt) -> bool:
    if isinstance(node, (ast.Return, ast.Raise)):
        return True
    if isinstance(node, ast.If):
        return _always_returns(node.body) and _always_returns(node.orelse)
    if isinstance(node, ast.Try):
        return _always_returns(node.finalbody) or (
            _always_returns(node.body)
            and bool(node.handlers)
            and all(_always_returns(handler.body) for handler in node.handlers)
            and (not node.orelse or _always_returns(node.orelse))
        )
    return False


def _always_returns(nodes: list[ast.stmt]) -> bool:
    return bool(nodes) and _stmt_always_returns(nodes[-1])


class StmtExceptionMixin:
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

    def _current_finally_return_context(self) -> dict[str, object] | None:
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
        seed_builder = getattr(expr_visitor, "_build_invoke_result_seed", None)
        if seed_builder is not None:
            expr_visitor.current_block = self.current_block
            value = seed_builder(return_type, loc)
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
        if not isinstance(signal_type, ir.Type) or not isinstance(yield_kind, str):
            raise RuntimeError("invalid finally return context")
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
        try_op: object,
        return_type: ir.Type,
        loc: ir.Location,
        *,
        always_returns: bool = False,
    ) -> None:
        if always_returns:
            self._emit_return_op(try_op.results[1], loc)
            return

        assert self.current_block is not None
        parent_region = self.current_block.region
        return_block = parent_region.blocks.append()
        merge_block = parent_region.blocks.append()
        with loc, self.insertion_point():
            cf_ops.CondBranchOp(try_op.results[0], [], [], return_block, merge_block)

        self._set_insertion_block(return_block)
        self._emit_return_op(try_op.results[1], loc)
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
                if not (
                    isinstance(signal_type, ir.Type)
                    and isinstance(yield_kind, str)
                    and isinstance(return_type, ir.Type)
                ):
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
                unsupported = _find_unsupported_finally_control(
                    [*node.body, *node.orelse, *node.finalbody]
                )
                for handler in node.handlers:
                    unsupported = unsupported or _find_unsupported_finally_control(
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
                if not try_terminated:
                    with loc, ir.InsertionPoint(try_end_block):
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
                if not finally_terminated:
                    with loc, ir.InsertionPoint(finally_end_block):
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
            finalbody_always_returns = bool(node.finalbody) and _always_returns(
                node.finalbody
            )
            if node.finalbody:
                unsupported = (
                    _find_unsupported_finally_control(
                        node.body, allow_raise=True, allow_return=True
                    )
                    or _find_unsupported_finally_control(
                        handler.body, allow_raise=True, allow_return=True
                    )
                    or _find_unsupported_finally_control(
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
                    _contains_return([*node.body, *handler.body])
                    or finalbody_always_returns
                )
            )
            result_types = (
                [i1, return_type] if returns_via_try else ([i1] if has_else else [])
            )
            with loc, self.insertion_point():
                try_op = py_ops.TryOp(result_types)

            with loc:
                try_block = try_op.try_region.blocks.append()
                except_block = try_op.except_region.blocks.append(
                    self.get_py_type("!py.exception")
                )
                if node.finalbody:
                    if returns_via_try:
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
            if returns_via_try:
                self._push_finally_return_context("try", i1)
            for stmt in node.body:
                self.visit(stmt)
            if returns_via_try:
                self._pop_finally_return_context()
            try_end_block = self.current_block
            try_terminated = try_end_block is None or self._block_terminated(
                try_end_block
            )
            self.pop_scope()
            if not try_terminated:
                with loc, ir.InsertionPoint(try_end_block):
                    if has_else:
                        completed = arith_ops.ConstantOp(
                            i1, ir.IntegerAttr.get(i1, 1)
                        ).result
                        py_ops.TryYieldOp([completed])
                    elif returns_via_try:
                        self._emit_finally_fallthrough_yield(
                            "try", return_type, i1, loc
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
            if returns_via_try:
                self._push_finally_return_context(
                    "except",
                    i1,
                    return_type=return_type,
                    swallow_raise=finalbody_always_returns,
                )
            try:
                for stmt in handler.body:
                    self.visit(stmt)
            finally:
                if returns_via_try:
                    self._pop_finally_return_context()
                self.pop_exception_context()
            except_end_block = self.current_block
            except_terminated = except_end_block is None or self._block_terminated(
                except_end_block
            )
            self.pop_scope()
            if not except_terminated:
                with loc, ir.InsertionPoint(except_end_block):
                    if has_else:
                        completed = arith_ops.ConstantOp(
                            i1, ir.IntegerAttr.get(i1, 0)
                        ).result
                        py_ops.ExceptYieldOp([completed])
                    elif returns_via_try:
                        self._emit_finally_fallthrough_yield(
                            "except", return_type, i1, loc
                        )
                    else:
                        py_ops.ExceptYieldOp([])

            if finally_block is not None:
                self._set_insertion_block(finally_block)
                self.push_scope()
                if returns_via_try and finalbody_always_returns:
                    self._push_finally_return_context("finally", i1)
                for stmt in node.finalbody:
                    self.visit(stmt)
                if returns_via_try and finalbody_always_returns:
                    self._pop_finally_return_context()
                finally_end_block = self.current_block
                finally_terminated = (
                    finally_end_block is None
                    or self._block_terminated(finally_end_block)
                )
                self.pop_scope()
                if not finally_terminated:
                    with loc, ir.InsertionPoint(finally_end_block):
                        operands = (
                            list(finally_block.arguments) if returns_via_try else []
                        )
                        py_ops.FinallyYieldOp(operands)

            self._set_insertion_block(prev_block)
            if returns_via_try:
                self._emit_finally_return_dispatch(
                    try_op,
                    return_type,
                    loc,
                    always_returns=finalbody_always_returns
                    or _stmt_always_returns(node),
                )
                return
            if has_else:
                assert self.current_block is not None
                parent_region = self.current_block.region
                else_block = parent_region.blocks.append()
                merge_block = parent_region.blocks.append()
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

        finalbody_always_returns = _always_returns(node.finalbody)
        unsupported = _find_unsupported_finally_control(
            node.body, allow_raise=True, allow_return=True
        ) or _find_unsupported_finally_control(
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
            _contains_return(node.body) or finalbody_always_returns
        )
        result_types = [i1, return_type] if finally_return else []
        with loc, self.insertion_point():
            try_op = py_ops.TryOp(result_types)

        with loc:
            try_block = try_op.try_region.blocks.append()
            if finally_return:
                finally_block = try_op.finally_region.blocks.append(*result_types)
            else:
                finally_block = try_op.finally_region.blocks.append()
        prev_block = self.current_block

        self._set_insertion_block(try_block)
        self.push_scope()
        if finally_return:
            self._push_finally_return_context(
                "try",
                i1,
                return_type=return_type,
                swallow_raise=finalbody_always_returns,
            )
        for stmt in node.body:
            self.visit(stmt)
        if finally_return:
            self._pop_finally_return_context()
        try_end_block = self.current_block
        try_terminated = try_end_block is None or self._block_terminated(try_end_block)
        self.pop_scope()
        if not try_terminated:
            with loc, ir.InsertionPoint(try_end_block):
                if finally_return:
                    self._emit_finally_fallthrough_yield("try", return_type, i1, loc)
                else:
                    py_ops.TryYieldOp([])

        self._set_insertion_block(finally_block)
        self.push_scope()
        if finally_return and finalbody_always_returns:
            self._push_finally_return_context("finally", i1)
        for stmt in node.finalbody:
            self.visit(stmt)
        if finally_return and finalbody_always_returns:
            self._pop_finally_return_context()
        finally_end_block = self.current_block
        finally_terminated = finally_end_block is None or self._block_terminated(
            finally_end_block
        )
        self.pop_scope()
        if not finally_terminated:
            with loc, ir.InsertionPoint(finally_end_block):
                operands = list(finally_block.arguments) if finally_return else []
                py_ops.FinallyYieldOp(operands)

        self._set_insertion_block(prev_block)
        if finally_return:
            self._emit_finally_return_dispatch(
                try_op,
                return_type,
                loc,
                always_returns=finalbody_always_returns,
            )

    def visit_TryStar(self, node: ast.TryStar) -> None:
        raise NotImplementedError("Try star statement not implemented")

    def visit_Assert(self, node: ast.Assert) -> None:
        raise NotImplementedError("Assert statement not implemented")
