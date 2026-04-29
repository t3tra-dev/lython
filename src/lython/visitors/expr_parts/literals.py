# pyright: reportAttributeAccessIssue=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false
from __future__ import annotations

import ast

from ...mlir import ir
from ...mlir.dialects import _lython_ops_gen as py_ops
from ...mlir.dialects import arith as arith_ops


class ExprLiteralMixin:
    """Expression lowering for constants and currently unsupported literal forms."""

    def visit_Set(self, node: ast.Set) -> None:
        raise NotImplementedError("Set literal not implemented")

    def visit_ListComp(self, node: ast.ListComp) -> None:
        raise NotImplementedError("List comprehension not implemented")

    def visit_Constant(self, node: ast.Constant) -> ir.Value:
        with self._loc(node), self.insertion_point():
            if node.value is None:
                return py_ops.NoneOp(self.get_py_type("!py.none")).result
            if isinstance(node.value, bool):
                result_type = self.get_py_type("!py.bool")
                prim_type = ir.IntegerType.get_signless(1, context=self.ctx)
                prim_attr = ir.IntegerAttr.get(prim_type, int(node.value))
                prim_value = arith_ops.ConstantOp(prim_type, prim_attr).result
                return py_ops.CastFromPrimOp(result_type, prim_value).result
            if isinstance(node.value, int):
                result_type = self.get_py_type("!py.int")
                return py_ops.IntConstantOp(result_type, str(node.value)).result
            if isinstance(node.value, float):
                result_type = self.get_py_type("!py.float")
                return py_ops.FloatConstantOp(result_type, node.value).result
            if isinstance(node.value, str):
                result_type = self.get_py_type("!py.str")
                attr = ir.StringAttr.get(node.value, self.ctx)
                return py_ops.StrConstantOp(result_type, attr).result
        raise NotImplementedError(f"Unsupported constant {node.value!r}")

    def visit_SetComp(self, node: ast.SetComp) -> None:
        raise NotImplementedError("Set comprehension not implemented")

    def visit_DictComp(self, node: ast.DictComp) -> None:
        raise NotImplementedError("Dict comprehension not implemented")

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        raise NotImplementedError("Generator expression not implemented")

    def visit_FormattedValue(self, node: ast.FormattedValue) -> None:
        raise NotImplementedError("Formatted value not implemented")

    def visit_JoinedStr(self, node: ast.JoinedStr) -> None:
        raise NotImplementedError("Joined string not implemented")

    def visit_Starred(self, node: ast.Starred) -> None:
        raise NotImplementedError("Starred expression not implemented")

    def visit_Slice(self, node: ast.Slice) -> None:
        raise NotImplementedError("Slice expression not implemented")
