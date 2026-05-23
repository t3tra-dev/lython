# pyright: reportAttributeAccessIssue=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false
from __future__ import annotations

import ast

from ...mlir import ir
from ...mlir.dialects import _lython_ops_gen as py_ops
from ...mlir.dialects import arith as arith_ops


class ExprNameMixin:
    """Expression lowering for names and attributes."""

    def visit_Name(self, node: ast.Name) -> ir.Value:
        if isinstance(node.ctx, ast.Store):
            raise NotImplementedError("Store context handled elsewhere")

        if self._in_native_func:
            prim_const = self.get_prim_constant(node.id)
            if prim_const is not None:
                mlir_type, value = prim_const
                with self._loc(node), self.insertion_point():
                    if isinstance(value, int):
                        attr = ir.IntegerAttr.get(mlir_type, value)
                    else:
                        attr = ir.FloatAttr.get(mlir_type, value)
                    return arith_ops.ConstantOp(mlir_type, attr).result

        for depth, scope in enumerate(reversed(self._scope_stack)):
            if node.id not in scope:
                continue
            value = scope[node.id]
            if depth == 0 or not str(value.type).startswith("!py.func<"):
                return value
            try:
                return self._clone_bound_callable_metadata(value, self._loc(node))
            except NotImplementedError:
                pass
            try:
                info = self.lookup_function_binding(node.id)
            except NameError:
                info = None
            if info is not None:
                return self._build_python_callable(
                    symbol=info.symbol,
                    func_type=value.type,
                    defaults=info.defaults,
                    kwdefaults=info.kwdefaults,
                    closure=info.closure,
                    loc=self._loc(node),
                    force_make_function=bool(
                        info.defaults is not None
                        or info.kwdefaults is not None
                        or info.closure is not None
                    ),
                )
            return self._clone_bound_callable_metadata(value, self._loc(node))
        try:
            func_info = self.lookup_function(node.id)
        except NameError as exc:
            raise NameError(f"Variable reference '{node.id}' not implemented") from exc
        if func_info.is_async:
            raise NotImplementedError(
                "Referencing async function values is not supported yet; "
                "call the async function or pass it to a supported asyncio builtin"
            )
        with self._loc(node), self.insertion_point():
            symbol = ir.FlatSymbolRefAttr.get(func_info.symbol, self.ctx)
            return py_ops.FuncObjectOp(func_info.func_type, symbol).result

    def visit_Attribute(self, node: ast.Attribute) -> ir.Value | None:
        obj = self.require_value(node.value, self.visit(node.value))
        result_type = self.get_attribute_type(obj.type, node.attr)

        with self._loc(node), self.insertion_point():
            return py_ops.AttrGetOp(result_type, obj, node.attr).result
