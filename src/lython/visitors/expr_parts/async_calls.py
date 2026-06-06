from __future__ import annotations

import ast
from typing import TYPE_CHECKING

from ...frontend.symbols import FunctionInfo
from ...mlir import ir
from ...mlir.dialects import _async_ops_gen as async_ops
from ...mlir.dialects import _lython_ops_gen as py_ops

if TYPE_CHECKING:
    from ..contracts import VisitorRuntime
else:
    VisitorRuntime = object


class ExprAsyncCallMixin(VisitorRuntime):
    """Async function calls and statically resolved asyncio builtins."""

    def _handle_async_function_call(
        self, node: ast.Call, func_info: FunctionInfo, loc: ir.Location
    ) -> ir.Value:
        coerced_args = self._build_async_function_call_args(node, func_info, loc)
        if len(func_info.result_types) != 1:
            raise NotImplementedError("Async functions must have one payload result")
        result_type = self.get_py_type(f"!py.coro<{func_info.result_types[0]}>")
        with loc, self.insertion_point():
            return py_ops.CoroCreateOp(
                result_type,
                ir.FlatSymbolRefAttr.get(func_info.symbol, self.ctx),
                coerced_args,
            ).result

    def _resolve_direct_async_call(
        self, node: ast.expr
    ) -> tuple[ast.Call, FunctionInfo] | None:
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            return None
        try:
            func_info = self.lookup_function(node.func.id)
        except NameError:
            return None
        if not func_info.is_async:
            return None
        return node, func_info

    def _build_async_function_call_args(
        self, node: ast.Call, func_info: FunctionInfo, loc: ir.Location
    ) -> list[ir.Value]:
        if node.keywords:
            raise NotImplementedError(
                "Keyword arguments for async functions are not supported yet"
            )
        if len(node.args) != len(func_info.arg_types):
            raise ValueError(
                f"Async function '{func_info.symbol}' expects {len(func_info.arg_types)} "
                f"arguments, got {len(node.args)}"
            )
        arg_values = [self.require_value(arg, self.visit(arg)) for arg in node.args]
        return [
            self.coerce_value_to_type(arg, expected, loc)
            for arg, expected in zip(arg_values, func_info.arg_types)
        ]

    def _emit_direct_async_call(
        self, node: ast.Call, func_info: FunctionInfo, loc: ir.Location
    ) -> ir.Value:
        coerced_args = self._build_async_function_call_args(node, func_info, loc)
        if len(func_info.result_types) != 1:
            raise NotImplementedError("Async functions must have one payload result")
        payload_type = func_info.result_types[0]
        async_value_type = ir.Type.parse(f"!async.value<{payload_type}>", self.ctx)
        with loc, self.insertion_point():
            return async_ops.CallOp(
                [async_value_type],
                ir.FlatSymbolRefAttr.get(func_info.symbol, self.ctx),
                coerced_args,
            ).operation.results[0]

    def _emit_immediate_async_call_await(
        self, node: ast.expr, loc: ir.Location
    ) -> ir.Value | None:
        resolved = self._resolve_direct_async_call(node)
        if resolved is None:
            return None
        call_node, func_info = resolved
        if func_info.maythrow:
            self._note_maythrow()
        awaitable = self._handle_async_function_call(call_node, func_info, loc)
        payload_type = self.get_awaitable_payload_type(awaitable.type)
        if payload_type is None:
            raise TypeError(
                f"Internal error: async call did not produce an awaitable, got {awaitable.type}"
            )
        with loc, self.insertion_point():
            return py_ops.AwaitOp(payload_type, awaitable).result

    def _resolve_asyncio_call(self, node: ast.expr, name: str) -> ast.Call | None:
        if not isinstance(node, ast.Call):
            return None
        if self._resolve_asyncio_builtin(node.func) != name:
            return None
        return node

    def _emit_asyncio_gather(self, node: ast.Call, loc: ir.Location) -> ir.Value:
        if node.keywords:
            raise NotImplementedError(
                "asyncio.gather keyword arguments are unsupported"
            )
        awaitables: list[ir.Value] = []
        payload_types: list[ir.Type] = []
        for arg in node.args:
            resolved = self._resolve_direct_async_call(arg)
            if resolved is not None:
                call_node, func_info = resolved
                if func_info.maythrow:
                    self._note_maythrow()
                awaitable = self._handle_async_function_call(call_node, func_info, loc)
            else:
                self._note_maythrow()
                awaitable = self.require_value(arg, self.visit(arg))
            payload_type = self.get_awaitable_payload_type(awaitable.type)
            if payload_type is None:
                raise TypeError(
                    "asyncio.gather expects statically typed awaitables, "
                    f"got {awaitable.type}"
                )
            awaitables.append(awaitable)
            payload_types.append(payload_type)

        if not awaitables:
            return self.build_tuple([], loc=loc)

        tuple_spec = ", ".join(str(payload_type) for payload_type in payload_types)
        tuple_type = self.get_py_type(f"!py.tuple<{tuple_spec}>")
        with loc, self.insertion_point():
            return py_ops.AsyncGatherOp(tuple_type, awaitables).result

    def _resolve_asyncio_builtin(self, func: ast.expr) -> str | None:
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            if self._static_modules.get(func.value.id) == "asyncio":
                if func.attr in {
                    "run",
                    "create_task",
                    "ensure_future",
                    "gather",
                    "sleep",
                }:
                    return func.attr
                raise NotImplementedError(f"asyncio.{func.attr} is not supported yet")
        if isinstance(func, ast.Name):
            binding = self._static_module_symbols.get(func.id)
            if binding is not None:
                module, name = binding
                if module == "asyncio":
                    return name
        return None

    def _handle_asyncio_builtin_call(
        self, name: str, node: ast.Call, loc: ir.Location
    ) -> ir.Value:
        if node.keywords:
            raise NotImplementedError(
                f"asyncio.{name} keyword arguments are unsupported"
            )

        if name == "run":
            if self.in_async_function():
                raise NotImplementedError(
                    "asyncio.run cannot be called from an async function"
                )
            if len(node.args) != 1:
                raise ValueError("asyncio.run expects exactly one awaitable")
            gather_call = self._resolve_asyncio_call(node.args[0], "gather")
            if gather_call is not None:
                return self._emit_asyncio_gather(gather_call, loc)
            immediate = self._emit_immediate_async_call_await(node.args[0], loc)
            if immediate is not None:
                return immediate
            awaitable = self.require_value(node.args[0], self.visit(node.args[0]))
            payload_type = self.get_awaitable_payload_type(awaitable.type)
            if payload_type is None:
                raise TypeError(
                    f"asyncio.run expects a statically typed awaitable, got {awaitable.type}"
                )
            with loc, self.insertion_point():
                return py_ops.AwaitOp(payload_type, awaitable).result

        if name in {"create_task", "ensure_future"}:
            if len(node.args) != 1:
                raise ValueError(f"asyncio.{name} expects exactly one coroutine")
            coroutine = self.require_value(node.args[0], self.visit(node.args[0]))
            payload_type = self.get_awaitable_payload_type(coroutine.type)
            if payload_type is None or not str(coroutine.type).startswith("!py.coro<"):
                raise TypeError(
                    f"asyncio.{name} expects a statically typed coroutine, "
                    f"got {coroutine.type}"
                )
            task_type = self.get_py_type(f"!py.task<{payload_type}>")
            with loc, self.insertion_point():
                return py_ops.TaskCreateOp(task_type, coroutine).result

        if name == "gather":
            raise NotImplementedError(
                "asyncio.gather must be immediately awaited or passed to asyncio.run"
            )

        if name == "sleep":
            if len(node.args) != 1:
                raise ValueError("asyncio.sleep expects exactly one duration")
            seconds = self.require_value(node.args[0], self.visit(node.args[0]))
            future_type = self.get_py_type("!py.future<!py.none>")
            with loc, self.insertion_point():
                return py_ops.AsyncSleepOp(future_type, seconds).result

        raise NotImplementedError(f"asyncio.{name} is not supported yet")
