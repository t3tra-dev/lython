from __future__ import annotations

import ast
from typing import TYPE_CHECKING

from ...frontend.locations import source_position
from ...frontend.symbols import FunctionInfo
from ...mlir import ir
from ...mlir.dialects import _lython_ops_gen as py_ops

if TYPE_CHECKING:
    from ..contracts import VisitorRuntime
else:
    VisitorRuntime = object


class _NestedFunctionCaptureCollector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.loaded_names: list[str] = []
        self._seen_loaded: set[str] = set()
        self.local_names: set[str] = set()

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load):
            if node.id not in self._seen_loaded:
                self.loaded_names.append(node.id)
                self._seen_loaded.add(node.id)
            return
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            self.local_names.add(node.id)

    def visit_arg(self, node: ast.arg) -> None:
        self.local_names.add(node.arg)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.local_names.add(node.name)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.local_names.add(node.name)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.local_names.add(node.name)

    def visit_Global(self, node: ast.Global) -> None:
        raise NotImplementedError(
            "global statement is not supported in nested functions"
        )

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        raise NotImplementedError(
            "nonlocal statement is not supported in nested functions"
        )


class _CapturedNameWriteCollector(ast.NodeVisitor):
    def __init__(self, names: set[str]) -> None:
        self._names = names
        self.writes: list[tuple[str, ast.AST]] = []

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, (ast.Store, ast.Del)) and node.id in self._names:
            self.writes.append((node.id, node))

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        return

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        return

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        return

    def visit_Lambda(self, node: ast.Lambda) -> None:
        return


class NestedFunctionMixin(VisitorRuntime):
    """Nested Python function lowering and capture analysis."""

    def _collect_nested_function_captures(
        self, node: ast.FunctionDef
    ) -> list[tuple[str, ir.Value]]:
        collector = _NestedFunctionCaptureCollector()
        for stmt in node.body:
            collector.visit(stmt)

        visible_symbols: dict[str, ir.Value] = {}
        for scope in reversed(self._scope_stack):
            for name, value in scope.items():
                if name not in visible_symbols:
                    visible_symbols[name] = value

        captures: list[tuple[str, ir.Value]] = []
        for name in collector.loaded_names:
            if name in collector.local_names:
                continue
            value = visible_symbols.get(name)
            if value is None:
                continue
            captures.append((name, value))
        return captures

    def _ensure_nested_capture_rebind_is_supported(
        self, node: ast.FunctionDef, captures: list[tuple[str, ir.Value]]
    ) -> None:
        if not captures:
            return
        enclosing = self.current_function_ast()
        if enclosing is None:
            return

        collector = _CapturedNameWriteCollector({name for name, _ in captures})
        for stmt in enclosing.body:
            collector.visit(stmt)

        def position(ast_node: ast.AST) -> tuple[int, int]:
            return source_position(ast_node) or (-1, -1)

        nested_pos = position(node)
        later_writes = sorted(
            {
                name
                for name, write_node in collector.writes
                if position(write_node) > nested_pos
            }
        )
        if later_writes:
            joined = ", ".join(later_writes)
            raise NotImplementedError(
                "Rebinding captured variables after nested function definition is "
                f"not supported yet: {joined}"
            )

    def _visit_nested_function_def(self, node: ast.FunctionDef) -> None:
        if node.decorator_list:
            raise NotImplementedError(
                "Decorators on nested functions are not supported"
            )
        self._validate_python_function_parameters(
            node, what=f"nested function '{node.name}'"
        )

        arg_type_specs = [
            self.annotation_to_py_type(arg.annotation) for arg in node.args.args
        ]
        kwonly_type_specs = [
            self.annotation_to_py_type(arg.annotation) for arg in node.args.kwonlyargs
        ]
        result_type_spec = self.annotation_to_py_type(node.returns)
        result_ir_type = self.get_py_type(result_type_spec)
        funcsig = self.build_funcsig(
            arg_type_specs, [result_type_spec], kwonly_types=kwonly_type_specs
        )
        py_func_sig = self.get_py_type(funcsig)
        py_func_type = self.get_py_type(f"!py.func<{funcsig}>")
        arg_name_attrs = [
            ir.StringAttr.get(arg.arg, self.ctx) for arg in node.args.args
        ]
        arg_names_attr = self.array_attr(arg_name_attrs) if arg_name_attrs else None
        kwonly_name_attrs = [
            ir.StringAttr.get(arg.arg, self.ctx) for arg in node.args.kwonlyargs
        ]
        kwonly_names_attr = (
            self.array_attr(kwonly_name_attrs) if kwonly_name_attrs else None
        )

        captures = self._collect_nested_function_captures(node)
        self._ensure_nested_capture_rebind_is_supported(node, captures)
        closure_types = [
            self.get_closure_storage_type(value.type) for _, value in captures
        ]
        closure_capture_arg_indices = [
            self.resolve_current_function_parameter_index_from_value(value)
            for _, value in captures
        ]
        closure_type_attrs = [
            ir.TypeAttr.get(ty, context=self.ctx) for ty in closure_types
        ]
        closure_types_attr = (
            self.array_attr(closure_type_attrs) if closure_type_attrs else None
        )

        symbol_name = self.next_nested_function_symbol(node.name)
        loc = self._loc(node)
        with loc, ir.InsertionPoint(self.module.body):
            func = py_ops.FuncOp(
                symbol_name,
                ir.TypeAttr.get(py_func_sig),
                arg_names=arg_names_attr,
                nothrow=True,
            )
            if kwonly_names_attr is not None:
                func.attributes["kwonly_names"] = kwonly_names_attr
            if closure_types_attr is not None:
                func.attributes["closure_types"] = closure_types_attr

        entry_arg_types = [self.get_py_type(spec) for spec in arg_type_specs]
        kwonly_arg_types = [self.get_py_type(spec) for spec in kwonly_type_specs]
        defaults_value, kwdefaults_value = self._build_python_function_metadata(
            node, entry_arg_types, kwonly_arg_types, loc
        )
        positional_default_callable_infos = self.collect_callable_default_infos(
            list(node.args.defaults)
        )
        kwonly_default_callable_infos = self.collect_callable_default_infos(
            list(node.args.kw_defaults)
        )
        all_entry_arg_types = entry_arg_types + kwonly_arg_types + closure_types
        with loc:
            if all_entry_arg_types:
                entry_block = func.body.blocks.append(*all_entry_arg_types)
            else:
                entry_block = func.body.blocks.append()

        prev_block = self.current_block
        self._set_insertion_block(entry_block)
        self.push_scope()
        self._enter_py_function(symbol_name)
        self.push_function_ast(node)
        self.push_return_type(result_ir_type)

        for arg, spec, value in zip(
            node.args.args, arg_type_specs, entry_block.arguments
        ):
            info = self.maybe_define_callable_parameter_binding(arg.arg, spec, value)
            if info is not None:
                value = self.annotate_known_callable_value(
                    value, info, loc=self._loc(arg)
                )
            self.define_symbol(arg.arg, value)
        kwonly_offset = len(entry_arg_types)
        for arg, spec, value in zip(
            node.args.kwonlyargs,
            kwonly_type_specs,
            entry_block.arguments[
                kwonly_offset : kwonly_offset + len(kwonly_arg_types)
            ],
        ):
            info = self.maybe_define_callable_parameter_binding(arg.arg, spec, value)
            if info is not None:
                value = self.annotate_known_callable_value(
                    value, info, loc=self._loc(arg)
                )
            self.define_symbol(arg.arg, value)
        for index, (capture_name, captured_outer_value) in enumerate(captures):
            capture_storage_value = entry_block.arguments[
                len(entry_arg_types) + len(kwonly_arg_types) + index
            ]
            capture_value = self.materialize_captured_value_from_storage(
                capture_storage_value,
                captured_outer_value.type,
                loc=loc,
            )
            try:
                capture_info = self.lookup_function_binding(capture_name)
            except NameError:
                capture_info = self.resolve_function_info_from_value(
                    captured_outer_value
                )
            if self.is_py_type(capture_value.type):
                with loc, self.insertion_point():
                    capture_value = py_ops.PublishOp(
                        capture_value.type, capture_value
                    ).result
            capture_value = self.annotate_known_callable_value(
                capture_value,
                capture_info,
                loc=loc,
            )
            self.define_symbol(capture_name, capture_value)
            if capture_info is not None:
                self.define_function_binding(capture_name, capture_info)

        for stmt in node.body:
            self.visit(stmt)

        active_block = self.current_block or entry_block
        if not self._block_terminated(active_block):
            if result_type_spec != "!py.none":
                raise NotImplementedError(
                    f"Nested function '{node.name}' must explicitly return {result_type_spec}"
                )
            with ir.Location.unknown(self.ctx), ir.InsertionPoint(active_block):
                none_val = py_ops.NoneOp(self.get_py_type("!py.none")).result
                py_ops.ReturnOp([none_val])

        maythrow, returned_function_info, returned_callable_arg_index = (
            self._exit_py_function()
        )
        self.pop_function_ast()
        self.pop_return_type()
        self._set_func_effect(func, maythrow)
        self.pop_scope()
        self._set_insertion_block(prev_block)

        with loc, self.insertion_point():
            closure = None
            if captures:
                closure = self.build_tuple(
                    [
                        self.materialize_closure_storage_value(value, loc=loc)
                        for _, value in captures
                    ],
                    loc=loc,
                )
            made_func = self._materialize_python_function_value(
                symbol_name,
                py_func_type,
                loc,
                defaults=defaults_value,
                kwdefaults=kwdefaults_value,
                closure=closure,
            )

        self.define_symbol(node.name, made_func)
        nested_info = FunctionInfo(
            symbol_name,
            py_func_type,
            tuple(entry_arg_types),
            (result_ir_type,),
            False,
            maythrow,
            tuple(arg.arg for arg in node.args.args),
            tuple(kwonly_arg_types),
            tuple(arg.arg for arg in node.args.kwonlyargs),
            tuple(
                arg.arg
                for arg, default in zip(node.args.kwonlyargs, node.args.kw_defaults)
                if default is not None
            ),
            len(node.args.defaults),
            positional_default_callable_infos,
            kwonly_default_callable_infos,
            defaults_value,
            kwdefaults_value,
            False,
            returned_function_info,
            returned_callable_arg_index,
            closure,
            tuple(closure_capture_arg_indices),
        )
        self.define_function_binding(node.name, nested_info)
        self._functions[symbol_name] = nested_info
