from __future__ import annotations

import ast
from collections.abc import Mapping
from typing import TYPE_CHECKING, NoReturn

from ..frontend.program import typed_program_for_module
from ..mlir import ir
from ..mlir.dialects import _lython_ops_gen as py_ops
from ..mlir.dialects import arith as arith_ops
from ._base import BaseVisitor

if TYPE_CHECKING:
    from .contracts import VisitorRuntime

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
        subvisitors: Mapping[str, VisitorRuntime],
    ) -> None:
        super().__init__(ctx, subvisitors=subvisitors)

    def visit_Module(self, node: ast.Module) -> None:
        """
        ```asdl
        Module(stmt* body, type_ignore* type_ignores)
        ```
        """
        module_name = "__main__"
        typed_program = typed_program_for_module(node)
        if typed_program is None:
            raise TypeError(
                "typed frontend analysis must run before py dialect emission"
            )
        self._set_typed_program(typed_program)
        with ir.Location.file("<module>", 1, 1, self.ctx):
            module = ir.Module.create()
        self._set_module(module)
        self._set_module_name(module_name)
        self._class_ast_defs = {
            stmt.name: stmt for stmt in node.body if isinstance(stmt, ast.ClassDef)
        }
        for visitor in self.subvisitors.values():
            visitor._class_ast_defs = self._class_ast_defs

        self.push_scope()
        with ir.InsertionPoint(module.body), ir.Location.unknown(self.ctx):
            base_exc_class = py_ops.ClassOp("BaseException")
            base_exc_class.body.blocks.append()
            exc_class = py_ops.ClassOp("Exception", base_names=["BaseException"])
            exc_class.body.blocks.append()
            base_exc_type = self.get_py_type('!py.class<"BaseException">')
            exc_type = self.get_py_type('!py.class<"Exception">')
            self.register_class("BaseException", base_exc_type, (), {}, {})
            self.register_class("Exception", exc_type, ("BaseException",), {}, {})

            builtin_sig = self.get_py_type(
                "!py.funcsig<[], vararg = !py.tuple<!py.str> -> [!py.none]>"
            )
            builtin_func_type = self.get_py_type(
                "!py.func<!py.funcsig<[], vararg = !py.tuple<!py.str> -> [!py.none]>>"
            )
            builtin = py_ops.FuncOp(
                "__builtin_print",
                ir.TypeAttr.get(builtin_sig),
                has_vararg=True,
                nothrow=True,
            )
            builtin_block = builtin.body.blocks.append(
                self.get_py_type("!py.tuple<!py.str>")
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
                nothrow=True,
            )
            main_block = main.body.blocks.append()

        self._set_insertion_block(main_block)
        self._enter_py_function("main")
        self.push_return_type(ir.IntegerType.get_signless(32, context=self.ctx))

        for stmt in node.body:
            self.visit(stmt)

        active_block = self.current_block or main_block
        if not self._block_terminated(active_block):
            with ir.InsertionPoint(active_block), ir.Location.unknown(self.ctx):
                i32 = ir.IntegerType.get_signless(32, context=self.ctx)
                zero = arith_ops.ConstantOp(i32, ir.IntegerAttr.get(i32, 0)).result
                py_ops.ReturnOp([zero])
        maythrow, _, _ = self._exit_py_function()
        self.pop_return_type()
        self._set_func_effect(main, maythrow)
        self.pop_scope()
        return None

    def visit_Interactive(self, node: ast.Interactive) -> NoReturn:
        """
        ```asdl
        Interactive(stmt* body)
        ```
        """
        raise NotImplementedError("Interactive mode not supported (static)")

    def visit_Expression(self, node: ast.Expression) -> NoReturn:
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
