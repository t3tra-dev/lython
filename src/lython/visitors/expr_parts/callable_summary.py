# pyright: reportAttributeAccessIssue=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false
from __future__ import annotations

import ast

from ...mlir import ir
from ...mlir.dialects import _lython_ops_gen as py_ops
from .._base import FunctionInfo


class ExprCallableSummaryMixin:
    def _attach_returned_callable_metadata(
        self,
        op: ir.Operation,
        func_info: FunctionInfo,
    ) -> None:
        self._attach_returned_callable_info(op, func_info.returned_function_info)

    def _attach_returned_callable_info(
        self,
        op: ir.Operation,
        returned: FunctionInfo | None,
    ) -> None:
        if returned is None:
            return
        op.attributes["ly.returned_callable_symbol"] = ir.FlatSymbolRefAttr.get(
            returned.symbol, self.ctx
        )
        op.attributes["ly.returned_callable_defaults_count"] = ir.IntegerAttr.get(
            ir.IntegerType.get_signless(64, context=self.ctx),
            returned.defaults_count,
        )
        if returned.kwdefault_names:
            op.attributes["ly.returned_callable_kwdefault_names"] = self.array_attr(
                [
                    ir.StringAttr.get(name, self.ctx)
                    for name in returned.kwdefault_names
                ],
            )

    def _materialize_known_callable_result(
        self,
        value: ir.Value,
        info: FunctionInfo | None,
        loc: ir.Location,
    ) -> ir.Value:
        if info is None or not str(value.type).startswith("!py.func<"):
            return value
        return self._build_python_callable(
            symbol=info.symbol,
            func_type=value.type,
            defaults=info.defaults,
            kwdefaults=info.kwdefaults,
            closure=info.closure,
            loc=loc,
            force_make_function=bool(
                info.defaults is not None
                or info.kwdefaults is not None
                or info.closure is not None
            ),
        )

    def _resolve_returned_callable_info_from_call(
        self,
        *,
        returned_function_info: FunctionInfo | None,
        returned_callable_arg_index: int | None,
        positional_param_names: tuple[str, ...] | list[str],
        kwonly_names: tuple[str, ...] | list[str],
        defaults_count: int = 0,
        positional_default_callable_infos: (
            tuple[FunctionInfo | None, ...] | list[FunctionInfo | None]
        ) = (),
        kwonly_default_callable_infos: (
            tuple[FunctionInfo | None, ...] | list[FunctionInfo | None]
        ) = (),
        positional_arg_nodes: list[ast.expr | None],
        positional_arg_values: list[ir.Value],
        keyword_arg_nodes: dict[str, ast.expr] | None = None,
        keyword_arg_values: dict[str, ir.Value] | None = None,
        loc: ir.Location,
    ) -> FunctionInfo | None:
        if returned_function_info is not None:
            return self._materialize_returned_callable_info_from_call(
                returned_function_info,
                positional_param_names=positional_param_names,
                kwonly_names=kwonly_names,
                positional_arg_values=positional_arg_values,
                keyword_arg_values=keyword_arg_values,
                loc=loc,
            )
        if returned_callable_arg_index is None:
            return None
        keyword_arg_nodes = keyword_arg_nodes or {}
        keyword_arg_values = keyword_arg_values or {}
        positional_param_names = list(positional_param_names)
        kwonly_names = list(kwonly_names)
        positional_default_callable_infos = list(positional_default_callable_infos)
        kwonly_default_callable_infos = list(kwonly_default_callable_infos)
        if returned_callable_arg_index < len(positional_arg_values):
            value = positional_arg_values[returned_callable_arg_index]
            node = (
                positional_arg_nodes[returned_callable_arg_index]
                if returned_callable_arg_index < len(positional_arg_nodes)
                else None
            )
            if node is None:
                return self.resolve_function_info_from_value(value)
            return self.resolve_function_info_from_expression(node, value)
        if returned_callable_arg_index < len(positional_param_names):
            param_name = positional_param_names[returned_callable_arg_index]
            if param_name in keyword_arg_values:
                value = keyword_arg_values[param_name]
                node = keyword_arg_nodes.get(param_name)
                if node is None:
                    return self.resolve_function_info_from_value(value)
                return self.resolve_function_info_from_expression(node, value)
            default_start = len(positional_param_names) - defaults_count
            if (
                defaults_count > 0
                and returned_callable_arg_index >= default_start
                and default_start >= 0
            ):
                default_index = returned_callable_arg_index - default_start
                if 0 <= default_index < len(positional_default_callable_infos):
                    return positional_default_callable_infos[default_index]
            return None
        else:
            kw_index = returned_callable_arg_index - len(positional_param_names)
            if kw_index < 0 or kw_index >= len(kwonly_names):
                return None
            param_name = kwonly_names[kw_index]
            if param_name in keyword_arg_values:
                value = keyword_arg_values[param_name]
                node = keyword_arg_nodes.get(param_name)
                if node is None:
                    return self.resolve_function_info_from_value(value)
                return self.resolve_function_info_from_expression(node, value)
            if 0 <= kw_index < len(kwonly_default_callable_infos):
                return kwonly_default_callable_infos[kw_index]
            return None

    def _resolve_argument_value_for_summary_index(
        self,
        *,
        parameter_index: int,
        positional_param_names: tuple[str, ...] | list[str],
        kwonly_names: tuple[str, ...] | list[str],
        positional_arg_values: list[ir.Value],
        keyword_arg_values: dict[str, ir.Value] | None = None,
    ) -> ir.Value | None:
        keyword_arg_values = keyword_arg_values or {}
        positional_param_names = list(positional_param_names)
        kwonly_names = list(kwonly_names)

        if parameter_index < len(positional_arg_values):
            return positional_arg_values[parameter_index]
        if parameter_index < len(positional_param_names):
            return keyword_arg_values.get(positional_param_names[parameter_index])
        kw_index = parameter_index - len(positional_param_names)
        if kw_index < 0 or kw_index >= len(kwonly_names):
            return None
        return keyword_arg_values.get(kwonly_names[kw_index])

    def _materialize_returned_callable_info_from_call(
        self,
        info: FunctionInfo,
        *,
        positional_param_names: tuple[str, ...] | list[str],
        kwonly_names: tuple[str, ...] | list[str],
        positional_arg_values: list[ir.Value],
        keyword_arg_values: dict[str, ir.Value] | None = None,
        loc: ir.Location,
    ) -> FunctionInfo:
        keyword_arg_values = keyword_arg_values or {}
        attempted_defaults = info.defaults is not None
        attempted_kwdefaults = info.kwdefaults is not None
        attempted_closure = info.closure is not None
        remapped_defaults = (
            self._remap_summary_value_to_callsite(
                info.defaults,
                summary_info=info,
                positional_param_names=positional_param_names,
                kwonly_names=kwonly_names,
                positional_arg_values=positional_arg_values,
                keyword_arg_values=keyword_arg_values,
                loc=loc,
            )
            if info.defaults is not None
            else None
        )
        remapped_kwdefaults = (
            self._remap_summary_value_to_callsite(
                info.kwdefaults,
                summary_info=info,
                positional_param_names=positional_param_names,
                kwonly_names=kwonly_names,
                positional_arg_values=positional_arg_values,
                keyword_arg_values=keyword_arg_values,
                loc=loc,
            )
            if info.kwdefaults is not None
            else None
        )
        remapped_closure = (
            self._remap_summary_value_to_callsite(
                info.closure,
                summary_info=info,
                positional_param_names=positional_param_names,
                kwonly_names=kwonly_names,
                positional_arg_values=positional_arg_values,
                keyword_arg_values=keyword_arg_values,
                loc=loc,
            )
            if info.closure is not None
            else None
        )
        remapped_or_rebuilt_closure = remapped_closure

        if remapped_or_rebuilt_closure is None and info.closure_capture_arg_indices:
            closure_values: list[ir.Value] = []
            for parameter_index in info.closure_capture_arg_indices:
                if parameter_index is None:
                    closure_values = []
                    break
                value = self._resolve_argument_value_for_summary_index(
                    parameter_index=parameter_index,
                    positional_param_names=positional_param_names,
                    kwonly_names=kwonly_names,
                    positional_arg_values=positional_arg_values,
                    keyword_arg_values=keyword_arg_values,
                )
                if value is None:
                    closure_values = []
                    break
                closure_values.append(value)
            if closure_values:
                remapped_or_rebuilt_closure = self.build_tuple(closure_values, loc=loc)

        if (
            remapped_defaults is not None
            or remapped_kwdefaults is not None
            or remapped_or_rebuilt_closure is not None
        ):
            return info._replace(
                defaults=remapped_defaults if attempted_defaults else None,
                kwdefaults=remapped_kwdefaults if attempted_kwdefaults else None,
                closure=(remapped_or_rebuilt_closure if attempted_closure else None),
            )
        return info

    def _build_python_callable(
        self,
        *,
        symbol: str,
        func_type: ir.Type,
        defaults: ir.Value | None,
        kwdefaults: ir.Value | None,
        closure: ir.Value | None,
        loc: ir.Location,
        force_make_function: bool = False,
    ) -> ir.Value:
        with loc, self.insertion_point():
            if (
                not force_make_function
                and defaults is None
                and kwdefaults is None
                and closure is None
            ):
                return py_ops.FuncObjectOp(func_type, symbol).result
            return py_ops.MakeFunctionOp(
                func_type,
                ir.FlatSymbolRefAttr.get(symbol, self.ctx),
                defaults=(
                    self._clone_bound_callable_metadata(defaults, loc)
                    if defaults is not None
                    else None
                ),
                kwdefaults=(
                    self._clone_bound_callable_metadata(kwdefaults, loc)
                    if kwdefaults is not None
                    else None
                ),
                closure=(
                    self._clone_bound_callable_metadata(closure, loc)
                    if closure is not None
                    else None
                ),
            ).result

    def _build_method_callable(
        self,
        *,
        symbol: str,
        func_type: ir.Type,
        defaults: ir.Value | None,
        kwdefaults: ir.Value | None,
        loc: ir.Location,
        force_make_function: bool = False,
    ) -> ir.Value:
        return self._build_python_callable(
            symbol=symbol,
            func_type=func_type,
            defaults=defaults,
            kwdefaults=kwdefaults,
            closure=None,
            loc=loc,
            force_make_function=force_make_function,
        )

    def _needs_keyword_callable_materialization(self, value: ir.Value) -> bool:
        current = value
        while True:
            op = self._value_operation(current)
            if op is None:
                return False
            op_name = str(getattr(op, "name", ""))
            if op_name == "py.publish":
                operands = list(getattr(op, "operands", ()))
                if not operands:
                    return False
                current = operands[0]
                continue
            return op_name == "py.func.object"
