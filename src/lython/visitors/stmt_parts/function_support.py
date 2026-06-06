from __future__ import annotations

import ast
from typing import TYPE_CHECKING

from ...mlir import ir
from ...mlir.dialects import _lython_ops_gen as py_ops
from ..models import NativeDecoratorInfo

if TYPE_CHECKING:
    from ..contracts import VisitorRuntime
else:
    VisitorRuntime = object


class FunctionSupportMixin(VisitorRuntime):
    """Shared helpers for Python function lowering."""

    def _get_native_decorator(
        self, decorators: list[ast.expr]
    ) -> NativeDecoratorInfo | None:
        for dec in decorators:
            if isinstance(dec, ast.Call) and isinstance(dec.func, ast.Name):
                if dec.func.id == "native" and "native" in self._lyrt_builtins:
                    result: NativeDecoratorInfo = {"gc": "none"}
                    for kw in dec.keywords:
                        if (
                            kw.arg == "gc"
                            and isinstance(kw.value, ast.Constant)
                            and isinstance(kw.value.value, str)
                        ):
                            result["gc"] = kw.value.value
                    return result
            if isinstance(dec, ast.Name) and dec.id == "native":
                if "native" in self._lyrt_builtins:
                    return {"gc": "none"}
        return None

    def _validate_python_function_parameters(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef, *, what: str
    ) -> None:
        if node.args.posonlyargs:
            raise NotImplementedError(
                f"Positional-only parameters are not supported in {what}"
            )
        if node.args.vararg is not None:
            raise NotImplementedError(f"*args is not supported in {what}")
        if node.args.kwarg is not None:
            raise NotImplementedError(f"**kwargs is not supported in {what}")

    def _build_python_function_metadata(
        self,
        node: ast.FunctionDef,
        positional_arg_types: list[ir.Type],
        kwonly_arg_types: list[ir.Type],
        loc: ir.Location,
    ) -> tuple[ir.Value | None, ir.Value | None]:
        defaults_value: ir.Value | None = None
        kwdefaults_value: ir.Value | None = None

        positional_defaults = node.args.defaults
        if positional_defaults:
            default_types = positional_arg_types[-len(positional_defaults) :]
            default_values: list[ir.Value] = []
            for default_node, expected_type in zip(positional_defaults, default_types):
                value = self.require_value(default_node, self.visit(default_node))
                value = self.coerce_value_to_type(
                    value, expected_type, self._loc(default_node)
                )
                default_values.append(value)
            defaults_value = self.build_tuple(default_values, loc=loc)

        kwdefault_entries = [
            (arg.arg, default_node, expected_type)
            for arg, default_node, expected_type in zip(
                node.args.kwonlyargs, node.args.kw_defaults, kwonly_arg_types
            )
            if default_node is not None
        ]
        if kwdefault_entries:
            value_type = kwdefault_entries[0][2]
            if any(
                str(expected) != str(value_type) for _, _, expected in kwdefault_entries
            ):
                raise NotImplementedError(
                    "Heterogeneous keyword-only defaults require typed tuple "
                    "metadata ABI"
                )
            dict_type = self.get_py_type(f"!py.dict<!py.str, {value_type}>")
            with loc, self.insertion_point():
                current = py_ops.DictEmptyOp(dict_type).result
                for name, default_node, expected_type in kwdefault_entries:
                    value = self.require_value(default_node, self.visit(default_node))
                    value = self.coerce_value_to_type(
                        value, expected_type, self._loc(default_node)
                    )
                    key = py_ops.StrConstantOp(
                        self.get_py_type("!py.str"), ir.StringAttr.get(name, self.ctx)
                    ).result
                    py_ops.DictInsertOp(current, key, value)
                kwdefaults_value = current

        return defaults_value, kwdefaults_value

    def _materialize_python_function_value(
        self,
        symbol_name: str,
        func_type: ir.Type,
        loc: ir.Location,
        *,
        defaults: ir.Value | None = None,
        kwdefaults: ir.Value | None = None,
        closure: ir.Value | None = None,
    ) -> ir.Value:
        with loc, self.insertion_point():
            if defaults is not None or kwdefaults is not None or closure is not None:
                return py_ops.MakeFunctionOp(
                    func_type,
                    ir.FlatSymbolRefAttr.get(symbol_name, self.ctx),
                    defaults=defaults,
                    kwdefaults=kwdefaults,
                    closure=closure,
                ).result
            return py_ops.FuncObjectOp(
                func_type, ir.FlatSymbolRefAttr.get(symbol_name, self.ctx)
            ).result
