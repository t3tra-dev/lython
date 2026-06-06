from __future__ import annotations

import ast
from typing import TYPE_CHECKING

from ...mlir import ir
from ...mlir.dialects import _lython_ops_gen as py_ops

if TYPE_CHECKING:
    from ..contracts import VisitorRuntime
else:
    VisitorRuntime = object


class ExprMiscMixin(VisitorRuntime):
    def visit_Lambda(self, node: ast.Lambda) -> None:
        """
        ラムダ式を処理する

        ```asdl
        Lambda(arguments args, expr body)
        ```
        """
        raise NotImplementedError("Lambda expression not implemented")

    def visit_IfExp(self, node: ast.IfExp) -> None:
        """
        三項演算子を処理する

        ```asdl
        IfExp(expr test, expr body, expr orelse)
        ```
        """
        raise NotImplementedError("If expression not implemented")

    def visit_Await(self, node: ast.Await) -> ir.Value:
        """
        await式を処理する

        ```asdl
        Await(expr value)
        ```
        """
        if not self.in_async_function():
            raise SyntaxError("'await' is only supported inside async functions")
        self._note_maythrow()
        gather_call = self._resolve_asyncio_call(node.value, "gather")
        if gather_call is not None:
            return self._emit_asyncio_gather(gather_call, self._loc(node))
        immediate = self._emit_immediate_async_call_await(node.value, self._loc(node))
        if immediate is not None:
            return immediate
        awaitable = self.require_value(node.value, self.visit(node.value))
        payload_type = self.typed_node_type(node)
        with self._loc(node), self.insertion_point():
            return py_ops.AwaitOp(payload_type, awaitable).result

    def visit_Yield(self, node: ast.Yield) -> None:
        """
        yield式を処理する

        ```asdl
        Yield(expr? value)
        ```
        """
        raise NotImplementedError("Yield expression not implemented")

    def visit_YieldFrom(self, node: ast.YieldFrom) -> None:
        """
        yield from式を処理する

        ```asdl
        YieldFrom(expr value)
        ```
        """
        raise NotImplementedError("YieldFrom expression not implemented")
