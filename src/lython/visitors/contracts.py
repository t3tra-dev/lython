from __future__ import annotations

import ast
from collections.abc import Sequence
from typing import NoReturn

from ..frontend.symbols import ClassInfo, FunctionInfo, MethodInfo
from ..frontend.types import TypeResolver
from ..mlir import ir
from ..mlir.dialects import _lython_ops_gen as py_ops
from .models import AttributeCarrier, FinallyReturnContext, TypedProgram, VisitResult


class VisitorRuntime:
    module: ir.Module
    ctx: ir.Context
    current_block: ir.Block | None
    subvisitors: dict[str, VisitorRuntime]
    _type_cache: dict[str, ir.Type]
    _scope_stack: list[dict[str, ir.Value]]
    _function_scope_stack: list[dict[str, FunctionInfo]]
    _module_name: str
    _functions: dict[str, FunctionInfo]
    _classes: dict[str, ClassInfo]
    _function_effect_stack: list[bool]
    _function_name_stack: list[str]
    _returned_function_info_stack: list[FunctionInfo | None]
    _returned_callable_arg_index_stack: list[int | None]
    _returned_function_info_valid_stack: list[bool]
    _callable_value_info: dict[int, FunctionInfo]
    _class_ast_defs: dict[str, ast.ClassDef]
    _function_ast_stack: list[ast.FunctionDef | ast.AsyncFunctionDef]
    _return_type_stack: list[ir.Type]
    _async_function_stack: list[bool]
    _finally_return_stack: list[FinallyReturnContext]
    _finally_return_stack_save: list[list[FinallyReturnContext] | None]
    _exception_context_stack: list[ir.Value]
    _exception_context_stack_save: list[list[ir.Value] | None]
    _nested_function_counter: int
    _current_class: str | None
    _current_class_definition_block: ir.Block | None
    _current_method: str | None
    _current_method_mutates_self: bool
    _pending_attributes: dict[str, ir.Type] | None
    _in_native_func: bool
    _native_gc_mode: str | None
    _prim_types: dict[str, str]
    _static_modules: dict[str, str]
    _static_module_symbols: dict[str, tuple[str, str]]
    _typed_program: TypedProgram | None
    _type_resolver: TypeResolver
    _lyrt_builtins: set[str]
    _prim_allocated: set[ir.Value]
    _prim_deallocated: set[ir.Value]
    _prim_alloc_sites: dict[ir.Value, ir.Location]
    _prim_constants: dict[str, tuple[ir.Type, int | float]]
    _pending_prim_const: tuple[ir.Type, int | float] | None

    def visit(self, node: ast.AST) -> VisitResult: ...
    def generic_visit(self, node: ast.AST) -> NoReturn: ...
    def require_value(self, node: ast.AST, result: object) -> ir.Value: ...
    def insertion_point(self) -> ir.InsertionPoint: ...
    def array_attr(self, attributes: Sequence[ir.Attribute]) -> ir.ArrayAttr: ...
    def get_py_type(self, type_spec: str) -> ir.Type: ...
    def _loc(self, node: ast.AST) -> ir.Location: ...

    def push_scope(self) -> None: ...
    def pop_scope(self) -> dict[str, ir.Value]: ...
    def current_scope(self) -> dict[str, ir.Value]: ...
    def define_symbol(self, name: str, value: ir.Value) -> None: ...
    def lookup_symbol(self, name: str) -> ir.Value: ...
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
    ) -> None: ...
    def define_function_binding(self, name: str, info: FunctionInfo) -> None: ...
    def undefine_function_binding(self, name: str) -> None: ...
    def lookup_function_binding(self, name: str) -> FunctionInfo: ...
    def lookup_function(self, name: str) -> FunctionInfo: ...
    def lookup_function_by_symbol(self, symbol: str) -> FunctionInfo: ...
    def register_class(
        self,
        name: str,
        class_type: ir.Type,
        base_names: tuple[str, ...],
        methods: dict[str, MethodInfo],
        attributes: dict[str, ir.Type] | None = None,
    ) -> None: ...
    def lookup_class(self, name: str) -> ClassInfo | None: ...
    def lookup_method_by_symbol(self, symbol: str) -> MethodInfo | None: ...
    def get_class_info_from_type(self, obj_type: ir.Type) -> ClassInfo | None: ...
    def get_list_element_type(self, list_type: ir.Type) -> ir.Type | None: ...
    def get_dict_key_value_types(
        self, dict_type: ir.Type
    ) -> tuple[ir.Type, ir.Type] | None: ...
    def get_attribute_type(self, obj_type: ir.Type, attr_name: str) -> ir.Type: ...
    def resolve_static_expression_type(self, expr: ast.expr) -> ir.Type | None: ...

    def annotation_to_static_class_type(
        self, annotation: ast.expr | None
    ) -> ir.Type: ...
    def annotation_to_py_type(self, annotation: ast.expr | None) -> str: ...
    def annotation_to_primitive_type(
        self, annotation: ast.expr | None
    ) -> ir.Type | None: ...
    def build_funcsig(
        self,
        arg_types: list[str],
        result_types: list[str],
        *,
        kwonly_types: list[str] | None = None,
        vararg_type: str | None = None,
        kwargs_type: str | None = None,
    ) -> str: ...
    def build_function_info_from_callable_type(
        self,
        name: str,
        func_type: ir.Type,
        *,
        maythrow: bool = True,
    ) -> FunctionInfo | None: ...

    def current_function_name(self) -> str | None: ...
    def current_function_ast(self) -> ast.FunctionDef | ast.AsyncFunctionDef | None: ...
    def is_nested_function_context(self) -> bool: ...
    def next_nested_function_symbol(self, lexical_name: str) -> str: ...
    def _enter_py_function(self, name: str) -> None: ...
    def _exit_py_function(self) -> tuple[bool, FunctionInfo | None, int | None]: ...
    def push_function_ast(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> None: ...
    def pop_function_ast(self) -> ast.FunctionDef | ast.AsyncFunctionDef: ...
    def _note_maythrow(self) -> None: ...
    def push_return_type(self, return_type: ir.Type) -> None: ...
    def pop_return_type(self) -> ir.Type: ...
    def current_return_type(self) -> ir.Type | None: ...
    def push_async_function(self, is_async: bool) -> None: ...
    def pop_async_function(self) -> bool: ...
    def in_async_function(self) -> bool: ...
    def get_awaitable_payload_type(self, awaitable_type: ir.Type) -> ir.Type | None: ...
    def push_exception_context(self, exception: ir.Value) -> None: ...
    def pop_exception_context(self) -> ir.Value: ...
    def current_exception_context(self) -> ir.Value | None: ...

    def build_tuple(
        self, values: list[ir.Value], *, loc: ir.Location | None = None
    ) -> ir.Value: ...
    def ensure_object(
        self, value: ir.Value, *, loc: ir.Location | None = None
    ) -> ir.Value: ...
    def coerce_value_to_type(
        self, value: ir.Value, expected_type: ir.Type, loc: ir.Location
    ) -> ir.Value: ...
    def coerce_operands_for_binary(
        self, lhs: ir.Value, rhs: ir.Value, loc: ir.Location
    ) -> tuple[ir.Value, ir.Value]: ...
    def get_closure_storage_type(self, value_type: ir.Type) -> ir.Type: ...
    def materialize_closure_storage_value(
        self, value: ir.Value, *, loc: ir.Location
    ) -> ir.Value: ...
    def materialize_captured_value_from_storage(
        self, storage_value: ir.Value, original_type: ir.Type, *, loc: ir.Location
    ) -> ir.Value: ...
    def copy_static_class_value(
        self, value: ir.Value, *, loc: ir.Location | None = None
    ) -> ir.Value: ...

    def collect_callable_default_infos(
        self, defaults: list[ast.expr | None] | tuple[ast.expr | None, ...]
    ) -> tuple[FunctionInfo | None, ...]: ...
    def resolve_function_info_from_value(
        self, value: ir.Value
    ) -> FunctionInfo | None: ...
    def resolve_function_info_from_expression(
        self, value_node: ast.expr, value: ir.Value
    ) -> FunctionInfo | None: ...
    def resolve_function_info_from_ast(
        self, value_node: ast.expr
    ) -> FunctionInfo | None: ...
    def resolve_current_function_parameter_index_from_value(
        self, value: ir.Value
    ) -> int | None: ...
    def resolve_current_function_parameter_index_from_expression(
        self, expr: ast.expr
    ) -> int | None: ...
    def maybe_define_callable_parameter_binding(
        self, name: str, type_spec: str, value: ir.Value
    ) -> FunctionInfo | None: ...
    def annotate_known_callable_value(
        self, value: ir.Value, info: FunctionInfo | None, *, loc: ir.Location
    ) -> ir.Value: ...
    def _attach_returned_callable_metadata(
        self, op: ir.Operation, func_info: FunctionInfo
    ) -> None: ...
    def _attach_returned_callable_info(
        self, op: ir.Operation, returned: FunctionInfo | None
    ) -> None: ...
    def _value_operation(self, value: ir.Value) -> ir.Operation | None: ...
    def _extract_make_function_operands(self, op: ir.Operation) -> tuple[
        ir.Value | None,
        ir.Value | None,
        ir.Value | None,
        ir.Value | None,
        ir.Value | None,
    ]: ...
    def _materialize_known_callable_result(
        self, value: ir.Value, info: FunctionInfo | None, loc: ir.Location
    ) -> ir.Value: ...
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
    ) -> ir.Value: ...
    def _build_method_callable(
        self,
        *,
        symbol: str,
        func_type: ir.Type,
        defaults: ir.Value | None,
        kwdefaults: ir.Value | None,
        loc: ir.Location,
        force_make_function: bool = False,
    ) -> ir.Value: ...
    def _needs_keyword_callable_materialization(self, value: ir.Value) -> bool: ...
    def _clone_bound_callable_metadata(
        self, value: ir.Value, loc: ir.Location
    ) -> ir.Value: ...

    def _validate_python_function_parameters(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef, *, what: str
    ) -> None: ...
    def _build_python_function_metadata(
        self,
        node: ast.FunctionDef,
        positional_arg_types: list[ir.Type],
        kwonly_arg_types: list[ir.Type],
        loc: ir.Location,
    ) -> tuple[ir.Value | None, ir.Value | None]: ...
    def _materialize_python_function_value(
        self,
        symbol_name: str,
        func_type: ir.Type,
        loc: ir.Location,
        *,
        defaults: ir.Value | None = None,
        kwdefaults: ir.Value | None = None,
        closure: ir.Value | None = None,
    ) -> ir.Value: ...

    def _enter_native_context(self, gc_mode: str) -> None: ...
    def _exit_native_context(self, loc: ir.Location) -> None: ...
    def _ensure_native_only(self, name: str, loc: ir.Location) -> None: ...
    def _ensure_not_in_native(self, name: str, loc: ir.Location) -> None: ...
    def _ensure_primitive_value(self, value: ir.Value, loc: ir.Location) -> None: ...
    def _check_prim_overwrite(self, name: str, loc: ir.Location) -> None: ...
    def register_prim_constant(
        self, name: str, mlir_type: ir.Type, value: int | float
    ) -> None: ...
    def get_prim_constant(self, name: str) -> tuple[ir.Type, int | float] | None: ...
    def get_primitive_type_from_spec(self, base_type: str, bits: int) -> ir.Type: ...
    def is_primitive_base_type(self, type_name: str) -> bool: ...
    def _handle_prim_constructor(
        self, node: ast.Call, base_type: str, loc: ir.Location
    ) -> ir.Value: ...
    def _handle_from_prim(self, node: ast.Call, loc: ir.Location) -> ir.Value: ...
    def _handle_to_prim_call(self, node: ast.Call, loc: ir.Location) -> ir.Value: ...
    def _handle_alloc_call(self, node: ast.Call, loc: ir.Location) -> ir.Value: ...
    def _handle_dealloc_call(self, node: ast.Call, loc: ir.Location) -> None: ...
    def _build_primitive_scalar(
        self, value: object, elem_type: ir.Type, loc: ir.Location
    ) -> ir.Value: ...
    def _build_linalg_fill(
        self, value: ir.Value, init: ir.Value, elem_type: ir.Type, loc: ir.Location
    ) -> ir.Value: ...
    def _build_linalg_matmul(
        self,
        lhs: ir.Value,
        rhs: ir.Value,
        init: ir.Value,
        elem_type: ir.Type,
        loc: ir.Location,
    ) -> ir.Value: ...
    def build_primitive_tensor_constructor(
        self,
        base_type: str,
        slice_expr: ast.expr,
        value_node: ast.AST,
        loc: ir.Location,
    ) -> ir.Value: ...
    def build_primitive_tensor_fill(
        self,
        base_type: str,
        slice_expr: ast.expr,
        fill_value: object,
        loc: ir.Location,
        shape_args: list[ast.expr] | None = None,
    ) -> ir.Value: ...
    def build_exception_value(
        self,
        *,
        message: ir.Value | None,
        loc: ir.Location,
        exc_type_name: str = "Exception",
        context: ir.Value | None = None,
    ) -> ir.Value: ...

    def _build_direct_vectorcall_operands(
        self,
        *,
        positional_args: list[ir.Value],
        keywords: list[ast.keyword],
        keyword_values: dict[str, ir.Value] | None = None,
        positional_param_types: tuple[ir.Type, ...] | list[ir.Type],
        positional_param_names: tuple[str, ...] | list[str],
        kwonly_param_types: tuple[ir.Type, ...] | list[ir.Type],
        kwonly_names: tuple[str, ...] | list[str],
        kwdefault_names: tuple[str, ...] | list[str],
        defaults_count: int,
        loc: ir.Location,
        leading_args: list[ir.Value] | None = None,
    ) -> tuple[ir.Value, ir.Value, ir.Value]: ...
    def _build_invoke_result_seed(
        self, result_type: ir.Type, loc: ir.Location
    ) -> ir.Value: ...
    def _emit_none_returning_invoke(
        self,
        callee: ir.Value,
        posargs: ir.Value,
        kwnames: ir.Value,
        kwvalues: ir.Value,
        loc: ir.Location,
    ) -> ir.Block: ...
    def _emit_value_returning_invoke(
        self,
        callee: ir.Value,
        posargs: ir.Value,
        kwnames: ir.Value,
        kwvalues: ir.Value,
        result_type: ir.Type,
        loc: ir.Location,
        returned_function_info: FunctionInfo | None = None,
    ) -> ir.Value: ...
    def _prepare_list_element_for_storage(
        self, value: ir.Value, element_type: ir.Type, loc: ir.Location
    ) -> ir.Value: ...
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
    ) -> FunctionInfo | None: ...
    def _resolve_argument_value_for_summary_index(
        self,
        *,
        parameter_index: int,
        positional_param_names: tuple[str, ...] | list[str],
        kwonly_names: tuple[str, ...] | list[str],
        positional_arg_values: list[ir.Value],
        keyword_arg_values: dict[str, ir.Value] | None = None,
    ) -> ir.Value | None: ...
    def _materialize_returned_callable_info_from_call(
        self,
        info: FunctionInfo,
        *,
        positional_param_names: tuple[str, ...] | list[str],
        kwonly_names: tuple[str, ...] | list[str],
        positional_arg_values: list[ir.Value],
        keyword_arg_values: dict[str, ir.Value] | None = None,
        loc: ir.Location,
    ) -> FunctionInfo: ...
    def _remap_summary_value_to_callsite(
        self,
        value: ir.Value,
        *,
        summary_info: FunctionInfo | None,
        positional_param_names: tuple[str, ...] | list[str],
        kwonly_names: tuple[str, ...] | list[str],
        positional_arg_values: list[ir.Value],
        keyword_arg_values: dict[str, ir.Value] | None,
        loc: ir.Location,
        cache: dict[int, ir.Value | None] | None = None,
    ) -> ir.Value | None: ...
    def _handle_class_instantiation(
        self, node: ast.Call, class_info: ClassInfo, loc: ir.Location
    ) -> ir.Value: ...
    def _handle_method_call(self, node: ast.Call, loc: ir.Location) -> ir.Value: ...
    def _handle_native_call(
        self, node: ast.Call, func_info: FunctionInfo, loc: ir.Location
    ) -> ir.Value: ...
    def _handle_async_function_call(
        self, node: ast.Call, func_info: FunctionInfo, loc: ir.Location
    ) -> ir.Value: ...
    def _resolve_direct_async_call(
        self, node: ast.expr
    ) -> tuple[ast.Call, FunctionInfo] | None: ...
    def _build_async_function_call_args(
        self, node: ast.Call, func_info: FunctionInfo, loc: ir.Location
    ) -> list[ir.Value]: ...
    def _emit_direct_async_call(
        self, node: ast.Call, func_info: FunctionInfo, loc: ir.Location
    ) -> ir.Value: ...
    def _emit_immediate_async_call_await(
        self, node: ast.expr, loc: ir.Location
    ) -> ir.Value | None: ...
    def _resolve_asyncio_call(self, node: ast.expr, name: str) -> ast.Call | None: ...
    def _emit_asyncio_gather(self, node: ast.Call, loc: ir.Location) -> ir.Value: ...
    def _resolve_asyncio_builtin(self, func: ast.expr) -> str | None: ...
    def _handle_asyncio_builtin_call(
        self, name: str, node: ast.Call, loc: ir.Location
    ) -> ir.Value: ...
    def _record_returned_callable_summary(
        self, info: FunctionInfo | None, arg_index: int | None
    ) -> None: ...
    def _emit_return_op(self, value: ir.Value, loc: ir.Location) -> None: ...
    def _emit_finally_fallthrough_yield(
        self,
        yield_kind: str,
        return_type: ir.Type,
        signal_type: ir.Type,
        loc: ir.Location,
    ) -> None: ...
    def _emit_finally_return_dispatch(
        self,
        try_op: py_ops.TryOp,
        return_type: ir.Type,
        loc: ir.Location,
        *,
        always_returns: bool = False,
    ) -> None: ...

    def _set_module(self, module: ir.Module) -> None: ...
    def _set_module_name(self, name: str) -> None: ...
    def _set_insertion_block(self, block: ir.Block | None) -> None: ...
    def _set_finally_return_stack(self, stack: list[FinallyReturnContext]) -> None: ...
    def _set_exception_context_stack(self, stack: list[ir.Value]) -> None: ...
    def _set_func_effect(self, func: AttributeCarrier, maythrow: bool) -> None: ...
    def _block_terminated(self, block: ir.Block) -> bool: ...
    def _advance_block_after_terminator(self) -> None: ...

    def _set_typed_program(self, typed_program: TypedProgram) -> None: ...
    def typed_node_type(self, node: ast.AST) -> ir.Type: ...
    def typed_node_type_or_none(self, node: ast.AST) -> ir.Type | None: ...

    def is_py_type(self, mlir_type: ir.Type) -> bool: ...
    def is_primitive_int_type(self, prim_type: ir.Type) -> bool: ...
    def is_primitive_float_type(self, prim_type: ir.Type) -> bool: ...
    def get_primitive_element_type(self, prim_type: ir.Type) -> ir.Type: ...
