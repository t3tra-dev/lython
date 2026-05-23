from __future__ import annotations

from typing import NamedTuple

from ..mlir import ir


class FunctionInfo(NamedTuple):
    symbol: str
    func_type: ir.Type
    arg_types: tuple[ir.Type, ...]
    result_types: tuple[ir.Type, ...]
    has_vararg: bool
    maythrow: bool
    arg_names: tuple[str, ...] = ()
    kwonly_arg_types: tuple[ir.Type, ...] = ()
    kwonly_names: tuple[str, ...] = ()
    kwdefault_names: tuple[str, ...] = ()
    defaults_count: int = 0
    positional_default_callable_infos: tuple["FunctionInfo | None", ...] = ()
    kwonly_default_callable_infos: tuple["FunctionInfo | None", ...] = ()
    defaults: ir.Value | None = None
    kwdefaults: ir.Value | None = None
    has_kwargs: bool = False
    returned_function_info: "FunctionInfo | None" = None
    returned_callable_arg_index: int | None = None
    closure: ir.Value | None = None
    closure_capture_arg_indices: tuple[int | None, ...] = ()
    is_async: bool = False


class MethodInfo(NamedTuple):
    """Information about a class method."""

    name: str
    arg_types: tuple[ir.Type, ...]
    result_types: tuple[ir.Type, ...]
    maythrow: bool
    mutates_self: bool
    init_method: bool
    arg_names: tuple[str, ...] = ()
    kwonly_arg_types: tuple[ir.Type, ...] = ()
    kwonly_names: tuple[str, ...] = ()
    kwdefault_names: tuple[str, ...] = ()
    defaults_count: int = 0
    positional_default_callable_infos: tuple["FunctionInfo | None", ...] = ()
    kwonly_default_callable_infos: tuple["FunctionInfo | None", ...] = ()
    defaults: ir.Value | None = None
    kwdefaults: ir.Value | None = None
    returned_function_info: "FunctionInfo | None" = None
    returned_callable_arg_index: int | None = None


class ClassInfo(NamedTuple):
    """Information about a class definition."""

    name: str
    class_type: ir.Type
    methods: dict[str, MethodInfo]
    attributes: dict[str, ir.Type]
