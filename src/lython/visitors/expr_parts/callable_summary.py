from __future__ import annotations

import ast
from typing import TYPE_CHECKING

from ...frontend.symbols import FunctionInfo
from ...mlir import ir

if TYPE_CHECKING:
    from ..contracts import VisitorRuntime
else:
    VisitorRuntime = object


class ExprCallableSummaryMixin(VisitorRuntime):
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
