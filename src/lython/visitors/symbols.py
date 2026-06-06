from __future__ import annotations

import ast
from typing import TYPE_CHECKING

from ..frontend.symbols import ClassInfo, FunctionInfo, MethodInfo
from ..mlir import ir

if TYPE_CHECKING:
    from .contracts import VisitorRuntime
else:
    VisitorRuntime = object


class SymbolTableMixin(VisitorRuntime):
    """IR-emission symbol, function, and class registries."""

    def push_scope(self) -> None:
        self._scope_stack.append({})
        self._function_scope_stack.append({})
        for visitor in self.subvisitors.values():
            visitor._scope_stack = self._scope_stack
            visitor._function_scope_stack = self._function_scope_stack

    def pop_scope(self) -> dict[str, ir.Value]:
        if not self._scope_stack:
            raise RuntimeError("Scope stack underflow")
        scope = self._scope_stack.pop()
        if not self._function_scope_stack:
            raise RuntimeError("Function scope stack underflow")
        self._function_scope_stack.pop()
        for visitor in self.subvisitors.values():
            visitor._scope_stack = self._scope_stack
            visitor._function_scope_stack = self._function_scope_stack
        return scope

    def current_scope(self) -> dict[str, ir.Value]:
        if not self._scope_stack:
            raise RuntimeError("Scope stack is empty")
        return self._scope_stack[-1]

    def define_symbol(self, name: str, value: ir.Value) -> None:
        self.current_scope()[name] = value

    def lookup_symbol(self, name: str) -> ir.Value:
        for scope in reversed(self._scope_stack):
            if name in scope:
                return scope[name]
        raise NameError(f"Undefined symbol '{name}'")

    def register_function(
        self,
        name: str,
        func_type: ir.Type,
        arg_types: list[ir.Type],
        result_types: list[ir.Type],
        *,
        symbol: str | None = None,
        has_vararg: bool = False,
        maythrow: bool = False,
        arg_names: list[str] | None = None,
        kwonly_arg_types: list[ir.Type] | None = None,
        kwonly_names: list[str] | None = None,
        kwdefault_names: list[str] | None = None,
        defaults_count: int = 0,
        positional_default_callable_infos: (
            tuple[FunctionInfo | None, ...] | list[FunctionInfo | None] | None
        ) = None,
        kwonly_default_callable_infos: (
            tuple[FunctionInfo | None, ...] | list[FunctionInfo | None] | None
        ) = None,
        defaults: ir.Value | None = None,
        kwdefaults: ir.Value | None = None,
        has_kwargs: bool = False,
        returned_function_info: FunctionInfo | None = None,
        returned_callable_arg_index: int | None = None,
        closure: ir.Value | None = None,
        closure_capture_arg_indices: (
            tuple[int | None, ...] | list[int | None] | None
        ) = None,
        is_async: bool = False,
    ) -> None:
        info = FunctionInfo(
            symbol or name,
            func_type,
            tuple(arg_types),
            tuple(result_types),
            has_vararg,
            maythrow,
            tuple(arg_names or ()),
            tuple(kwonly_arg_types or ()),
            tuple(kwonly_names or ()),
            tuple(kwdefault_names or ()),
            defaults_count,
            tuple(positional_default_callable_infos or ()),
            tuple(kwonly_default_callable_infos or ()),
            defaults,
            kwdefaults,
            has_kwargs,
            returned_function_info,
            returned_callable_arg_index,
            closure,
            tuple(closure_capture_arg_indices or ()),
            is_async,
        )
        self._functions[name] = info

    def define_function_binding(self, name: str, info: FunctionInfo) -> None:
        if not self._function_scope_stack:
            raise RuntimeError("Function scope stack is empty")
        self._function_scope_stack[-1][name] = info

    def undefine_function_binding(self, name: str) -> None:
        if not self._function_scope_stack:
            raise RuntimeError("Function scope stack is empty")
        self._function_scope_stack[-1].pop(name, None)

    def lookup_function_binding(self, name: str) -> FunctionInfo:
        for scope in reversed(self._function_scope_stack):
            if name in scope:
                return scope[name]
        raise NameError(f"Unresolved bound function '{name}'")

    def lookup_function(self, name: str) -> FunctionInfo:
        if name not in self._functions:
            raise NameError(f"Unresolved function '{name}'")
        return self._functions[name]

    def lookup_function_by_symbol(self, symbol: str) -> FunctionInfo:
        for scope in reversed(self._function_scope_stack):
            for info in scope.values():
                if info.symbol == symbol:
                    return info
        for info in self._functions.values():
            if info.symbol == symbol:
                return info
        raise NameError(f"Unresolved function symbol '{symbol}'")

    def _lookup_symbol_type_or_none(self, name: str) -> ir.Type | None:
        try:
            return self.lookup_symbol(name).type
        except NameError:
            return None

    def _lookup_function_binding_or_none(self, name: str) -> FunctionInfo | None:
        try:
            return self.lookup_function_binding(name)
        except NameError:
            return None

    def _lookup_function_or_none(self, name: str) -> FunctionInfo | None:
        try:
            return self.lookup_function(name)
        except NameError:
            return None

    def register_class(
        self,
        name: str,
        class_type: ir.Type,
        base_names: tuple[str, ...],
        methods: dict[str, MethodInfo],
        attributes: dict[str, ir.Type] | None = None,
    ) -> None:
        info = ClassInfo(name, class_type, base_names, methods, attributes or {})
        self._classes[name] = info

    def lookup_class(self, name: str) -> ClassInfo | None:
        return self._classes.get(name)

    def lookup_method_by_symbol(self, symbol: str) -> MethodInfo | None:
        for class_info in self._classes.values():
            for method in class_info.methods.values():
                if f"{class_info.name}.{method.name}" == symbol:
                    return method
        return None

    def get_class_info_from_type(self, obj_type: ir.Type) -> ClassInfo | None:
        return self._type_resolver.class_info_from_type(obj_type)

    def get_list_element_type(self, list_type: ir.Type) -> ir.Type | None:
        return self._type_resolver.list_element_type(list_type)

    def get_dict_key_value_types(
        self, dict_type: ir.Type
    ) -> tuple[ir.Type, ir.Type] | None:
        return self._type_resolver.dict_key_value_types(dict_type)

    def get_attribute_type(self, obj_type: ir.Type, attr_name: str) -> ir.Type:
        stmt_visitor = self.subvisitors.get("Stmt") or self
        return self._type_resolver.attribute_type(
            obj_type,
            attr_name,
            pending_attributes=stmt_visitor._pending_attributes,
        )

    def resolve_static_expression_type(self, expr: ast.expr) -> ir.Type | None:
        return self.typed_node_type(expr)
