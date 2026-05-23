from __future__ import annotations

import ast
from typing import TYPE_CHECKING

from ...frontend.symbols import FunctionInfo
from ...mlir import ir
from ...mlir.dialects import func as func_ops
from ..models import NativeDecoratorInfo

if TYPE_CHECKING:
    from ..contracts import VisitorRuntime
else:
    VisitorRuntime = object


class NativeFunctionMixin(VisitorRuntime):
    """Lowering for @native primitive functions."""

    def _visit_native_function_def(
        self, node: ast.FunctionDef, native_info: NativeDecoratorInfo
    ) -> None:
        arg_types: list[ir.Type] = []
        for arg in node.args.args:
            prim_type = self.annotation_to_primitive_type(arg.annotation)
            if prim_type is None:
                raise ValueError(
                    f"@native function '{node.name}' argument '{arg.arg}' "
                    f"must have a primitive type annotation"
                )
            arg_types.append(prim_type)

        result_type = self.annotation_to_primitive_type(node.returns)
        if result_type is None:
            raise ValueError(
                f"@native function '{node.name}' must have a primitive return type annotation"
            )
        result_types = [result_type]

        loc = self._loc(node)
        func_type = ir.FunctionType.get(arg_types, result_types, context=self.ctx)
        with loc, ir.InsertionPoint(self.module.body):
            func = func_ops.FuncOp(node.name, func_type)
            func.attributes["native"] = ir.UnitAttr.get(self.ctx)

        self._register_native_function(node.name, arg_types, result_types)

        with loc:
            if arg_types:
                entry_block = func.body.blocks.append(*arg_types)
            else:
                entry_block = func.body.blocks.append()

        prev_block = self.current_block
        self._set_insertion_block(entry_block)
        self.push_scope()
        self.push_return_type(result_type)
        self.push_function_ast(node)
        self._enter_native_context(native_info.get("gc", "none"))

        for arg, value in zip(node.args.args, entry_block.arguments):
            self.define_symbol(arg.arg, value)

        for stmt in node.body:
            self.visit(stmt)

        active_block = self.current_block or entry_block
        if not self._block_terminated(active_block):
            raise NotImplementedError(
                f"@native function '{node.name}' must explicitly return"
            )

        self._exit_native_context(loc)
        self.pop_function_ast()
        self.pop_return_type()
        self.pop_scope()
        self._set_insertion_block(prev_block)

    def _register_native_function(
        self,
        name: str,
        arg_types: list[ir.Type],
        result_types: list[ir.Type],
    ) -> None:
        func_type = ir.FunctionType.get(arg_types, result_types, context=self.ctx)
        info = FunctionInfo(
            symbol=name,
            func_type=func_type,
            arg_types=tuple(arg_types),
            result_types=tuple(result_types),
            has_vararg=False,
            maythrow=False,
        )
        self._functions[name] = info
