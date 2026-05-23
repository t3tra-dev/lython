from __future__ import annotations

import ast
from typing import TYPE_CHECKING

from ...frontend.symbols import FunctionInfo
from ...mlir import ir
from ..mlir_access import op_attributes, op_name, op_operands, value_argument_index

if TYPE_CHECKING:
    from ..contracts import VisitorRuntime
else:
    VisitorRuntime = object


class CallableResolutionMixin(VisitorRuntime):
    """Resolve callable summaries from values, AST expressions, and parameters."""

    def collect_callable_default_infos(
        self, defaults: list[ast.expr | None] | tuple[ast.expr | None, ...]
    ) -> tuple[FunctionInfo | None, ...]:
        infos: list[FunctionInfo | None] = []
        for default in defaults:
            if default is None:
                infos.append(None)
            else:
                infos.append(self.resolve_function_info_from_ast(default))
        return tuple(infos)

    def resolve_function_info_from_value(self, value: ir.Value) -> FunctionInfo | None:
        if not str(value.type).startswith("!py.func<"):
            return None

        current = value
        while True:
            info = self._callable_value_info.get(id(current))
            if info is not None:
                return info

            op = self._value_operation(current)
            if op is None:
                return None

            name = op_name(op)
            attributes = op_attributes(op)
            returned_symbol_attr = (
                attributes["ly.returned_callable_symbol"]
                if "ly.returned_callable_symbol" in attributes
                else None
            )
            if returned_symbol_attr is not None:
                symbol = str(returned_symbol_attr)
                if symbol.startswith("@"):
                    symbol = symbol[1:]
                try:
                    return self.lookup_function_by_symbol(symbol)
                except NameError:
                    return None
            if name == "py.publish":
                operands = op_operands(op)
                if not operands:
                    return None
                current = operands[0]
                continue

            if name in {"py.func.object", "py.make_function"}:
                target_attr = attributes["target"] if "target" in attributes else None
                if target_attr is None:
                    return None
                symbol = str(target_attr)
                if symbol.startswith("@"):
                    symbol = symbol[1:]
                try:
                    info = self.lookup_function_by_symbol(symbol)
                except NameError:
                    return None
                if name == "py.func.object":
                    return info
                defaults, kwdefaults, closure, _, _ = (
                    self._extract_make_function_operands(op)
                )
                return info._replace(
                    func_type=value.type,
                    defaults=defaults,
                    kwdefaults=kwdefaults,
                    closure=closure,
                )

            return None

    def resolve_function_info_from_expression(
        self, value_node: ast.expr, value: ir.Value
    ) -> FunctionInfo | None:
        value_info = self.resolve_function_info_from_value(value)
        ast_info = self.resolve_function_info_from_ast(value_node)
        if value_info is None:
            return ast_info
        if ast_info is None:
            return value_info
        if value_info.symbol != ast_info.symbol:
            return value_info
        return value_info._replace(
            func_type=ast_info.func_type or value_info.func_type,
            arg_types=ast_info.arg_types or value_info.arg_types,
            result_types=ast_info.result_types or value_info.result_types,
            has_vararg=ast_info.has_vararg or value_info.has_vararg,
            maythrow=ast_info.maythrow or value_info.maythrow,
            arg_names=ast_info.arg_names or value_info.arg_names,
            kwonly_arg_types=ast_info.kwonly_arg_types or value_info.kwonly_arg_types,
            kwonly_names=ast_info.kwonly_names or value_info.kwonly_names,
            kwdefault_names=ast_info.kwdefault_names or value_info.kwdefault_names,
            defaults_count=ast_info.defaults_count or value_info.defaults_count,
            positional_default_callable_infos=(
                ast_info.positional_default_callable_infos
                or value_info.positional_default_callable_infos
            ),
            kwonly_default_callable_infos=(
                ast_info.kwonly_default_callable_infos
                or value_info.kwonly_default_callable_infos
            ),
            defaults=(
                value_info.defaults
                if value_info.defaults is not None
                else ast_info.defaults
            ),
            kwdefaults=(
                value_info.kwdefaults
                if value_info.kwdefaults is not None
                else ast_info.kwdefaults
            ),
            has_kwargs=ast_info.has_kwargs or value_info.has_kwargs,
            returned_function_info=(
                ast_info.returned_function_info
                if ast_info.returned_function_info is not None
                else value_info.returned_function_info
            ),
            returned_callable_arg_index=(
                ast_info.returned_callable_arg_index
                if ast_info.returned_callable_arg_index is not None
                else value_info.returned_callable_arg_index
            ),
            closure=(
                value_info.closure
                if value_info.closure is not None
                else ast_info.closure
            ),
            closure_capture_arg_indices=(
                value_info.closure_capture_arg_indices
                or ast_info.closure_capture_arg_indices
            ),
        )

    def _select_argument_expression_for_callable_summary(
        self,
        call_node: ast.Call,
        *,
        returned_callable_arg_index: int | None,
        positional_param_names: tuple[str, ...] | list[str],
        kwonly_names: tuple[str, ...] | list[str],
        leading_exprs: list[ast.expr] | None = None,
    ) -> ast.expr | None:
        if returned_callable_arg_index is None:
            return None

        positional_param_names = list(positional_param_names)
        kwonly_names = list(kwonly_names)
        leading_exprs = list(leading_exprs or [])
        positional_exprs = [*leading_exprs, *list(call_node.args)]

        if returned_callable_arg_index < len(positional_exprs):
            return positional_exprs[returned_callable_arg_index]

        if returned_callable_arg_index < len(positional_param_names):
            param_name = positional_param_names[returned_callable_arg_index]
        else:
            kw_index = returned_callable_arg_index - len(positional_param_names)
            if kw_index < 0 or kw_index >= len(kwonly_names):
                return None
            param_name = kwonly_names[kw_index]

        for keyword in call_node.keywords:
            if keyword.arg == param_name:
                return keyword.value
        return None

    def _resolve_omitted_default_callable_info(
        self,
        *,
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
    ) -> FunctionInfo | None:
        if returned_callable_arg_index is None:
            return None

        positional_param_names = list(positional_param_names)
        kwonly_names = list(kwonly_names)
        positional_default_callable_infos = list(positional_default_callable_infos)
        kwonly_default_callable_infos = list(kwonly_default_callable_infos)

        if returned_callable_arg_index < len(positional_param_names):
            default_start = len(positional_param_names) - defaults_count
            if (
                defaults_count <= 0
                or default_start < 0
                or returned_callable_arg_index < default_start
            ):
                return None
            default_index = returned_callable_arg_index - default_start
            if 0 <= default_index < len(positional_default_callable_infos):
                return positional_default_callable_infos[default_index]
            return None

        kw_index = returned_callable_arg_index - len(positional_param_names)
        if kw_index < 0 or kw_index >= len(kwonly_names):
            return None
        if 0 <= kw_index < len(kwonly_default_callable_infos):
            return kwonly_default_callable_infos[kw_index]
        return None

    def resolve_function_info_from_ast(
        self, value_node: ast.expr
    ) -> FunctionInfo | None:

        if isinstance(value_node, ast.Name):
            try:
                return self.lookup_function_binding(value_node.id)
            except NameError:
                try:
                    return self.lookup_function(value_node.id)
                except NameError:
                    return None

        if isinstance(value_node, ast.Call) and isinstance(value_node.func, ast.Name):
            try:
                callee_info = self.lookup_function_binding(value_node.func.id)
            except NameError:
                try:
                    callee_info = self.lookup_function(value_node.func.id)
                except NameError:
                    callee_info = None
            if callee_info is not None:
                if callee_info.returned_function_info is not None:
                    return callee_info.returned_function_info
                selected = self._select_argument_expression_for_callable_summary(
                    value_node,
                    returned_callable_arg_index=callee_info.returned_callable_arg_index,
                    positional_param_names=callee_info.arg_names,
                    kwonly_names=callee_info.kwonly_names,
                )
                if selected is not None:
                    return self.resolve_function_info_from_ast(selected)
                return self._resolve_omitted_default_callable_info(
                    returned_callable_arg_index=callee_info.returned_callable_arg_index,
                    positional_param_names=callee_info.arg_names,
                    kwonly_names=callee_info.kwonly_names,
                    defaults_count=callee_info.defaults_count,
                    positional_default_callable_infos=callee_info.positional_default_callable_infos,
                    kwonly_default_callable_infos=callee_info.kwonly_default_callable_infos,
                )

        if isinstance(value_node, ast.Call) and isinstance(
            value_node.func, ast.Attribute
        ):
            if (
                isinstance(value_node.func.value, ast.Name)
                and value_node.func.value.id in self._static_modules
            ):
                return None
            receiver_type = self.resolve_static_expression_type(value_node.func.value)
            if receiver_type is not None:
                class_info = self.get_class_info_from_type(receiver_type)
                if class_info is not None:
                    method_info = class_info.methods.get(value_node.func.attr)
                    if method_info is not None:
                        if method_info.returned_function_info is not None:
                            return method_info.returned_function_info
                        selected = self._select_argument_expression_for_callable_summary(
                            value_node,
                            returned_callable_arg_index=method_info.returned_callable_arg_index,
                            positional_param_names=method_info.arg_names,
                            kwonly_names=method_info.kwonly_names,
                            leading_exprs=[value_node.func.value],
                        )
                        if selected is not None:
                            return self.resolve_function_info_from_ast(selected)
                        return self._resolve_omitted_default_callable_info(
                            returned_callable_arg_index=method_info.returned_callable_arg_index,
                            positional_param_names=method_info.arg_names,
                            kwonly_names=method_info.kwonly_names,
                            defaults_count=method_info.defaults_count,
                            positional_default_callable_infos=method_info.positional_default_callable_infos,
                            kwonly_default_callable_infos=method_info.kwonly_default_callable_infos,
                        )

        return None

    def resolve_current_function_parameter_index_from_value(
        self, value: ir.Value
    ) -> int | None:
        return value_argument_index(value)

    def resolve_current_function_parameter_index_from_expression(
        self, expr: ast.expr
    ) -> int | None:
        if isinstance(expr, ast.Name):
            current = self.current_function_ast()
            if current is None:
                return None
            for index, arg in enumerate(current.args.args):
                if arg.arg == expr.id:
                    return index
            kwonly_offset = len(current.args.args)
            for index, arg in enumerate(current.args.kwonlyargs):
                if arg.arg == expr.id:
                    return kwonly_offset + index
            return None

        if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name):
            try:
                callee_info = self.lookup_function_binding(expr.func.id)
            except NameError:
                try:
                    callee_info = self.lookup_function(expr.func.id)
                except NameError:
                    callee_info = None
            if callee_info is None:
                return None
            selected = self._select_argument_expression_for_callable_summary(
                expr,
                returned_callable_arg_index=callee_info.returned_callable_arg_index,
                positional_param_names=callee_info.arg_names,
                kwonly_names=callee_info.kwonly_names,
            )
            if selected is None:
                return None
            return self.resolve_current_function_parameter_index_from_expression(
                selected
            )

        if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Attribute):
            if (
                isinstance(expr.func.value, ast.Name)
                and expr.func.value.id in self._static_modules
            ):
                return None
            receiver_type = self.resolve_static_expression_type(expr.func.value)
            if receiver_type is None:
                return None
            class_info = self.get_class_info_from_type(receiver_type)
            if class_info is None:
                return None
            method_info = class_info.methods.get(expr.func.attr)
            if method_info is None:
                return None
            selected = self._select_argument_expression_for_callable_summary(
                expr,
                returned_callable_arg_index=method_info.returned_callable_arg_index,
                positional_param_names=method_info.arg_names,
                kwonly_names=method_info.kwonly_names,
                leading_exprs=[expr.func.value],
            )
            if selected is None:
                return None
            return self.resolve_current_function_parameter_index_from_expression(
                selected
            )

        return None
