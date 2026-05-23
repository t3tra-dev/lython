from __future__ import annotations

import ast
from typing import TYPE_CHECKING

from ...mlir import ir
from ...mlir.dialects import _async_ops_gen as async_ops
from ...mlir.dialects import _lython_ops_gen as py_ops

if TYPE_CHECKING:
    from ..contracts import VisitorRuntime
else:
    VisitorRuntime = object


class AsyncFunctionMixin(VisitorRuntime):
    """Lowering for top-level async Python function definitions."""

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        if self._in_native_func:
            raise NotImplementedError(
                "Nested async functions inside @native are not supported"
            )
        if self.is_nested_function_context():
            raise NotImplementedError("Nested async functions are not supported yet")
        if node.decorator_list:
            raise NotImplementedError("Decorators on async functions are not supported")

        self._validate_python_function_parameters(
            node, what=f"async function '{node.name}'"
        )
        if node.args.kwonlyargs:
            raise NotImplementedError(
                "Keyword-only parameters on async functions are not supported yet"
            )
        if node.args.defaults:
            raise NotImplementedError(
                "Default parameters on async functions are not supported yet"
            )

        arg_type_specs = [
            self.annotation_to_py_type(arg.annotation) for arg in node.args.args
        ]
        result_type_spec = self.annotation_to_py_type(node.returns)
        entry_arg_types = [self.get_py_type(spec) for spec in arg_type_specs]
        result_ir_type = self.get_py_type(result_type_spec)
        async_result_type = ir.Type.parse(f"!async.value<{result_type_spec}>", self.ctx)
        exception_cell_type = ir.Type.parse("!llvm.ptr", self.ctx)
        coro_type = self.get_py_type(f"!py.coro<{result_type_spec}>")
        symbol_name = node.name if node.name != "main" else "__lython_async_main"

        loc = self._loc(node)
        func_type = ir.FunctionType.get(
            [*entry_arg_types, exception_cell_type],
            [async_result_type],
            context=self.ctx,
        )
        with loc, ir.InsertionPoint(self.module.body):
            func = async_ops.FuncOp(symbol_name, ir.TypeAttr.get(func_type))

        self.register_function(
            node.name,
            coro_type,
            entry_arg_types,
            [result_ir_type],
            symbol=symbol_name,
            maythrow=False,
            arg_names=[arg.arg for arg in node.args.args],
            is_async=True,
        )

        with loc:
            entry_block = func.body.blocks.append(*entry_arg_types, exception_cell_type)

        prev_block = self.current_block
        self._set_insertion_block(entry_block)
        self.push_scope()
        self._enter_py_function(symbol_name)
        self.push_function_ast(node)
        self.push_return_type(result_ir_type)
        self.push_async_function(True)

        for arg, spec, value in zip(
            node.args.args, arg_type_specs, entry_block.arguments
        ):
            info = self.maybe_define_callable_parameter_binding(arg.arg, spec, value)
            if info is not None:
                value = self.annotate_known_callable_value(
                    value, info, loc=self._loc(arg)
                )
            self.define_symbol(arg.arg, value)

        for stmt in node.body:
            self.visit(stmt)

        active_block = self.current_block or entry_block
        if not self._block_terminated(active_block):
            if result_type_spec != "!py.none":
                raise NotImplementedError(
                    f"Async function '{node.name}' must explicitly return {result_type_spec}"
                )
            with ir.Location.unknown(self.ctx), ir.InsertionPoint(active_block):
                none_val = py_ops.NoneOp(self.get_py_type("!py.none")).result
                async_ops.ReturnOp([none_val])

        maythrow, returned_function_info, returned_callable_arg_index = (
            self._exit_py_function()
        )
        self.pop_async_function()
        self.pop_function_ast()
        self.pop_return_type()
        self.register_function(
            node.name,
            coro_type,
            entry_arg_types,
            [result_ir_type],
            symbol=symbol_name,
            maythrow=maythrow,
            arg_names=[arg.arg for arg in node.args.args],
            returned_function_info=returned_function_info,
            returned_callable_arg_index=returned_callable_arg_index,
            is_async=True,
        )
        self.pop_scope()
        self._set_insertion_block(prev_block)
