from __future__ import annotations

import ast
from typing import Any

from ..mlir import ir
from ..mlir.dialects import _lython_ops_gen as py_ops
from ..mlir.dialects import arith as arith_ops
from ._base import BaseVisitor

__all__ = ["ModVisitor"]


class ModVisitor(BaseVisitor):
    """
    モジュール(ソースファイル全体)を訪問するクラス
    Python の ast.Module に対応

    ```asdl
    mod = Module(stmt* body, type_ignore* type_ignores)
        | Interactive(stmt* body)
        | Expression(expr body)
        | FunctionType(expr* argtypes, expr returns)
    ```
    """

    def __init__(
        self,
        ctx: ir.Context,
        *,
        subvisitors: dict[str, BaseVisitor],
    ) -> None:
        super().__init__(ctx, subvisitors=subvisitors)

    def visit_Module(self, node: ast.Module) -> None:
        """
        ```asdl
        Module(stmt* body, type_ignore* type_ignores)
        ```
        """
        module_name = getattr(node, "lython_module_name", "__main__")
        with ir.Location.file("<module>", 1, 1, self.ctx):
            module = ir.Module.create()
        self._set_module(module)
        self._set_module_name(module_name)

        self.push_scope()
        with ir.InsertionPoint(module.body), ir.Location.unknown(self.ctx):
            builtin_sig = self.get_py_type(
                "!py.funcsig<[], vararg = !py.tuple<!py.object> -> [!py.none]>"
            )
            builtin_func_type = self.get_py_type(
                "!py.func<!py.funcsig<[], vararg = !py.tuple<!py.object> -> [!py.none]>>"
            )
            builtin = py_ops.FuncOp(
                "__builtin_print",
                ir.TypeAttr.get(builtin_sig),
                has_vararg=True,
            )
            builtin_block = builtin.body.blocks.append(
                self.get_py_type("!py.tuple<!py.object>")
            )
            with ir.InsertionPoint(builtin_block), ir.Location.unknown(self.ctx):
                none_value = py_ops.NoneOp(self.get_py_type("!py.none")).result
                py_ops.ReturnOp([none_value])
            self.register_function(
                "print",
                builtin_func_type,
                [],
                [self.get_py_type("!py.none")],
                symbol="__builtin_print",
                has_vararg=True,
            )

            main_sig = self.get_py_type("!py.funcsig<[] -> [i32]>")
            main = py_ops.FuncOp(
                "main",
                ir.TypeAttr.get(main_sig),
            )
            main_block = main.body.blocks.append()

        self._set_insertion_block(main_block)

        for stmt in node.body:
            self.visit(stmt)

        with ir.InsertionPoint(main_block), ir.Location.unknown(self.ctx):
            i32 = ir.IntegerType.get_signless(32)
            zero = arith_ops.ConstantOp(i32, ir.IntegerAttr.get(i32, 0)).result
            py_ops.ReturnOp([zero])
        self.pop_scope()
        return None

    def visit_Interactive(self, node: ast.Interactive) -> Any:
        """
        ```asdl
        Interactive(stmt* body)
        ```
        """
        raise NotImplementedError("Interactive mode not supported (static)")

    def visit_Expression(self, node: ast.Expression) -> Any:
        """
        ```asdl
        Expression(expr body)
        ```
        """
        raise NotImplementedError("Expression mode not supported")

    def visit_FunctionType(self, node: ast.FunctionType) -> None:
        """
        ```asdl
        FunctionType(expr* argtypes, expr returns)
        ```
        """
        raise NotImplementedError("Function type not supported in this static compiler")
