from __future__ import annotations

import ast
from typing import Any

from ..mlir import ir
from ..mlir.dialects import _lython_ops_gen as py_ops
from ..mlir.dialects import arith as arith_ops
from ..mlir.dialects import cf as cf_ops
from ..mlir.dialects import func as func_ops
from ._base import PRIMITIVE_BASE_TYPES, BaseVisitor, FunctionInfo, MethodInfo

__all__ = ["StmtVisitor"]


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
        raise NotImplementedError("global statement is not supported in nested functions")

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        raise NotImplementedError("nonlocal statement is not supported in nested functions")


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


class StmtVisitor(BaseVisitor):
    """
    文(stmt)ノードを訪問し、
    MLIRを生成するクラス。

    ```asdl
    stmt = FunctionDef(identifier name, arguments args,
                       stmt* body, expr* decorator_list, expr? returns,
                       string? type_comment, type_param* type_params)
          | AsyncFunctionDef(identifier name, arguments args,
                             stmt* body, expr* decorator_list, expr? returns,
                             string? type_comment, type_param* type_params)

          | ClassDef(identifier name,
             expr* bases,
             keyword* keywords,
             stmt* body,
             expr* decorator_list,
             type_param* type_params)
          | Return(expr? value)

          | Delete(expr* targets)
          | Assign(expr* targets, expr value, string? type_comment)
          | TypeAlias(expr name, type_param* type_params, expr value)
          | AugAssign(expr target, operator op, expr value)
          -- 'simple' indicates that we annotate simple name without parens
          | AnnAssign(expr target, expr annotation, expr? value, int simple)

          -- use 'orelse' because else is a keyword in target languages
          | For(expr target, expr iter, stmt* body, stmt* orelse, string? type_comment)
          | AsyncFor(expr target, expr iter, stmt* body, stmt* orelse, string? type_comment)
          | While(expr test, stmt* body, stmt* orelse)
          | If(expr test, stmt* body, stmt* orelse)
          | With(withitem* items, stmt* body, string? type_comment)
          | AsyncWith(withitem* items, stmt* body, string? type_comment)

          | Match(expr subject, match_case* cases)

          | Raise(expr? exc, expr? cause)
          | Try(stmt* body, excepthandler* handlers, stmt* orelse, stmt* finalbody)
          | TryStar(stmt* body, excepthandler* handlers, stmt* orelse, stmt* finalbody)
          | Assert(expr test, expr? msg)

          | Import(alias* names)
          | ImportFrom(identifier? module, alias* names, int? level)

          | Global(identifier* names)
          | Nonlocal(identifier* names)
          | Expr(expr value)
          | Pass | Break | Continue

          -- col_offset is the byte offset in the utf8 string the parser uses
          attributes (int lineno, int col_offset, int? end_lineno, int? end_col_offset)
    ```
    """

    def __init__(
        self,
        ctx: ir.Context,
        *,
        subvisitors: dict[str, BaseVisitor],
    ) -> None:
        super().__init__(ctx, subvisitors=subvisitors)

    def _get_native_decorator(
        self, decorators: list[ast.expr]
    ) -> dict[str, Any] | None:
        """
        Check if function has @native decorator and extract its arguments.
        Returns dict with gc mode or None if not a native function.
        """
        for dec in decorators:
            # @native(gc="none") is ast.Call
            if isinstance(dec, ast.Call) and isinstance(dec.func, ast.Name):
                if dec.func.id == "native" and "native" in self._lyrt_builtins:
                    result: dict[str, Any] = {"gc": "none"}  # default
                    for kw in dec.keywords:
                        if kw.arg == "gc" and isinstance(kw.value, ast.Constant):
                            result["gc"] = kw.value.value
                    return result
            # @native without parens is ast.Name
            if isinstance(dec, ast.Name) and dec.id == "native":
                if "native" in self._lyrt_builtins:
                    return {"gc": "none"}
        return None

    def _validate_python_function_parameters(
        self, node: ast.FunctionDef, *, what: str
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
                value = self.coerce_value_to_type(value, expected_type, self._loc(default_node))
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
            dict_type = self.get_py_type("!py.dict<!py.str, !py.object>")
            with loc, self.insertion_point():
                current = py_ops.DictEmptyOp(dict_type).result
                for name, default_node, expected_type in kwdefault_entries:
                    value = self.require_value(default_node, self.visit(default_node))
                    value = self.coerce_value_to_type(
                        value, expected_type, self._loc(default_node)
                    )
                    value = self.ensure_object(value, loc=loc)
                    key = py_ops.StrConstantOp(
                        self.get_py_type("!py.str"), ir.StringAttr.get(name, self.ctx)
                    ).result
                    current = py_ops.DictInsertOp(dict_type, current, key, value).result
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

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """
        関数定義:
        def hoge(n: int) -> int:
            ...
        などを受け取り、静的型のIRを生成する

        ```asdl
        FunctionDef(
            identifier name,
            arguments args,
            stmt* body,
            expr* decorator_list,
            expr? returns,
            string? type_comment,
            type_param* type_params
        )
        ```
        """
        if self._in_native_func:
            raise NotImplementedError("Nested Python functions inside @native are not supported")
        if self.is_nested_function_context():
            return self._visit_nested_function_def(node)

        # Check for @native decorator
        native_info = self._get_native_decorator(node.decorator_list)
        if native_info is not None:
            return self._visit_native_function_def(node, native_info)

        self._validate_python_function_parameters(node, what=f"function '{node.name}'")

        # Regular Python function
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
        arg_names_attr = (
            ir.ArrayAttr.get(  # pyright: ignore[reportUnknownMemberType]
                arg_name_attrs, context=self.ctx
            )
            if arg_name_attrs
            else None
        )
        kwonly_name_attrs = [
            ir.StringAttr.get(arg.arg, self.ctx) for arg in node.args.kwonlyargs
        ]
        kwonly_names_attr = (
            ir.ArrayAttr.get(kwonly_name_attrs, context=self.ctx)
            if kwonly_name_attrs
            else None
        )
        loc = self._loc(node)
        with loc, ir.InsertionPoint(self.module.body):
            func = py_ops.FuncOp(
                node.name,
                ir.TypeAttr.get(py_func_sig),
                arg_names=arg_names_attr,
                nothrow=True,
            )
            if kwonly_names_attr is not None:
                func.attributes["kwonly_names"] = kwonly_names_attr
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
        if defaults_value is not None or kwdefaults_value is not None:
            bound_func = self._materialize_python_function_value(
                node.name,
                py_func_type,
                loc,
                defaults=defaults_value,
                kwdefaults=kwdefaults_value,
            )
            self.define_symbol(node.name, bound_func)
        self.register_function(
            node.name,
            py_func_type,
            entry_arg_types,
            [result_ir_type],
            maythrow=False,
            arg_names=[arg.arg for arg in node.args.args],
            kwonly_arg_types=kwonly_arg_types,
            kwonly_names=[arg.arg for arg in node.args.kwonlyargs],
            kwdefault_names=[
                arg.arg
                for arg, default in zip(node.args.kwonlyargs, node.args.kw_defaults)
                if default is not None
            ],
            defaults_count=len(node.args.defaults),
            positional_default_callable_infos=positional_default_callable_infos,
            kwonly_default_callable_infos=kwonly_default_callable_infos,
            defaults=defaults_value,
            kwdefaults=kwdefaults_value,
        )
        with loc:
            all_entry_arg_types = entry_arg_types + kwonly_arg_types
            if all_entry_arg_types:
                entry_block = func.body.blocks.append(*all_entry_arg_types)
            else:
                entry_block = func.body.blocks.append()
        prev_block = self.current_block
        self._set_insertion_block(entry_block)
        self.push_scope()
        self._enter_py_function(node.name)
        self.push_function_ast(node)
        self.push_return_type(result_ir_type)
        for arg, spec, value in zip(node.args.args, arg_type_specs, entry_block.arguments):
            info = self.maybe_define_callable_parameter_binding(arg.arg, spec, value)
            if info is not None:
                value = self.annotate_known_callable_value(
                    value, info, loc=self._loc(arg)
                )
            self.define_symbol(arg.arg, value)
        offset = len(node.args.args)
        for arg, spec, value in zip(
            node.args.kwonlyargs, kwonly_type_specs, entry_block.arguments[offset:]
        ):
            info = self.maybe_define_callable_parameter_binding(arg.arg, spec, value)
            if info is not None:
                value = self.annotate_known_callable_value(
                    value, info, loc=self._loc(arg)
                )
            self.define_symbol(arg.arg, value)
        if defaults_value is not None or kwdefaults_value is not None:
            with loc, self.insertion_point():
                self.define_symbol(
                    node.name,
                    py_ops.FuncObjectOp(
                        py_func_type, ir.FlatSymbolRefAttr.get(node.name, self.ctx)
                    ).result,
                )
        for stmt in node.body:
            self.visit(stmt)
        active_block = self.current_block or entry_block
        if not self._block_terminated(active_block):
            if result_type_spec != "!py.none":
                raise NotImplementedError(
                    f"Function '{node.name}' must explicitly return {result_type_spec}"
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
        self.register_function(
            node.name,
            py_func_type,
            entry_arg_types,
            [result_ir_type],
            maythrow=maythrow,
            arg_names=[arg.arg for arg in node.args.args],
            kwonly_arg_types=kwonly_arg_types,
            kwonly_names=[arg.arg for arg in node.args.kwonlyargs],
            kwdefault_names=[
                arg.arg
                for arg, default in zip(node.args.kwonlyargs, node.args.kw_defaults)
                if default is not None
            ],
            defaults_count=len(node.args.defaults),
            positional_default_callable_infos=positional_default_callable_infos,
            kwonly_default_callable_infos=kwonly_default_callable_infos,
            defaults=defaults_value,
            kwdefaults=kwdefaults_value,
            returned_function_info=returned_function_info,
            returned_callable_arg_index=returned_callable_arg_index,
        )
        self.pop_scope()
        self._set_insertion_block(prev_block)

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
            return (
                int(getattr(ast_node, "lineno", -1)),
                int(getattr(ast_node, "col_offset", -1)),
            )

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
            raise NotImplementedError("Decorators on nested functions are not supported")
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
        arg_names_attr = (
            ir.ArrayAttr.get(arg_name_attrs, context=self.ctx)
            if arg_name_attrs
            else None
        )
        kwonly_name_attrs = [
            ir.StringAttr.get(arg.arg, self.ctx) for arg in node.args.kwonlyargs
        ]
        kwonly_names_attr = (
            ir.ArrayAttr.get(kwonly_name_attrs, context=self.ctx)
            if kwonly_name_attrs
            else None
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
            ir.ArrayAttr.get(closure_type_attrs, context=self.ctx)
            if closure_type_attrs
            else None
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

        for arg, spec, value in zip(node.args.args, arg_type_specs, entry_block.arguments):
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
            entry_block.arguments[kwonly_offset : kwonly_offset + len(kwonly_arg_types)],
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
                capture_info = self.resolve_function_info_from_value(captured_outer_value)
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
                for arg, default in zip(
                    node.args.kwonlyargs, node.args.kw_defaults
                )
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

    def _visit_native_function_def(
        self, node: ast.FunctionDef, native_info: dict[str, Any]
    ) -> None:
        """
        Generate func.func for @native decorated functions.

        Native functions operate in the Primitive World:
        - Use MLIR primitive types (i8, i32, f64, etc.)
        - Generate arith.* operations instead of py.num.*
        - No GC involvement
        """
        # Get primitive types for arguments
        arg_types: list[ir.Type] = []
        for arg in node.args.args:
            prim_type = self.annotation_to_primitive_type(arg.annotation)
            if prim_type is None:
                raise ValueError(
                    f"@native function '{node.name}' argument '{arg.arg}' "
                    f"must have a primitive type annotation"
                )
            arg_types.append(prim_type)

        # Get primitive return type
        result_type = self.annotation_to_primitive_type(node.returns)
        if result_type is None:
            raise ValueError(
                f"@native function '{node.name}' must have a primitive return type annotation"
            )
        result_types = [result_type]

        loc = self._loc(node)

        # Create func.func operation with 'native' attribute
        # This attribute marks functions operating in the Primitive World (𝒫)
        # and enables static verification that no py.* types are used inside
        func_type = ir.FunctionType.get(arg_types, result_types, context=self.ctx)
        with loc, ir.InsertionPoint(self.module.body):
            func = func_ops.FuncOp(node.name, func_type)
            func.attributes["native"] = ir.UnitAttr.get(self.ctx)

        # Register the function with primitive types
        # Note: We use a special marker to indicate this is a native function
        self._register_native_function(node.name, arg_types, result_types)

        # Create entry block
        with loc:
            if arg_types:
                entry_block = func.body.blocks.append(*arg_types)
            else:
                entry_block = func.body.blocks.append()

        prev_block = self.current_block
        self._set_insertion_block(entry_block)
        self.push_scope()
        self.push_return_type(result_type)
        self.push_function_ast(node)

        # Enter native mode
        self._enter_native_context(native_info.get("gc", "none"))

        # Register arguments in scope
        for arg, value in zip(node.args.args, entry_block.arguments):
            self.define_symbol(arg.arg, value)

        # Process function body
        for stmt in node.body:
            self.visit(stmt)

        # Check for missing return
        active_block = self.current_block or entry_block
        if not self._block_terminated(active_block):
            raise NotImplementedError(
                f"@native function '{node.name}' must explicitly return"
            )

        # Exit native mode (includes allocation checks)
        self._exit_native_context(loc)
        self.pop_function_ast()
        self.pop_return_type()

        self.pop_scope()
        self._set_insertion_block(prev_block)

    def _register_native_function(
        self,
        name: str,
        arg_types: list[ir.Type],
        result_types: list[ir.Type],
    ) -> None:
        """Register a native function in the function table."""
        from ._base import FunctionInfo

        # Create a fake py.func type for compatibility
        # Native functions are identified by their types being primitives
        func_type = ir.FunctionType.get(arg_types, result_types, context=self.ctx)
        info = FunctionInfo(
            symbol=name,
            func_type=func_type,
            arg_types=tuple(arg_types),
            result_types=tuple(result_types),
            has_vararg=False,
            maythrow=False,
        )
        self._functions[name] = info

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """
        非同期関数の定義を処理する

        ```asdl
        AsyncFunctionDef(
            identifier name,
            arguments args,
            stmt* body,
            expr* decorator_list,
            expr? returns,
            string? type_comment,
            type_param* type_params
        )
        ```
        """
        raise NotImplementedError("Async function definition not supported")

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """
        クラス定義

        ```asdl
        ClassDef(
            identifier name,
            expr* bases,
            keyword* keywords,
            stmt* body,
            expr* decorator_list,
            type_param* type_params
        )
        ```
        """
        if node.bases:
            raise NotImplementedError("Class inheritance not yet supported")
        if node.decorator_list:
            raise NotImplementedError("Class decorators not yet supported")

        class_name = node.name
        loc = self._loc(node)

        # Create py.class operation
        with loc, ir.InsertionPoint(self.module.body):
            class_op = py_ops.ClassOp(class_name)

        # Create a block in the class body for methods
        class_body_block = class_op.body.blocks.append()

        # Register the class type (class name must be quoted in type syntax)
        class_type = self.get_py_type(f'!py.class<"{class_name}">')

        # Process class body - only method definitions for now
        prev_block = self.current_block
        self._set_insertion_block(class_body_block)
        self.push_scope()

        # Set current class context for method processing
        prev_class = getattr(self, "_current_class", None)
        self._current_class = class_name
        prev_class_definition_block = getattr(
            self, "_current_class_definition_block", None
        )
        self._current_class_definition_block = prev_block

        # Track instance attributes and their types during class processing
        prev_pending_attrs = getattr(self, "_pending_attributes", None)
        self._pending_attributes: dict[str, ir.Type] = {}

        # Collect method info
        methods: dict[str, MethodInfo] = {}
        method_defs = [stmt for stmt in node.body if isinstance(stmt, ast.FunctionDef)]
        ordered_methods = [
            stmt for stmt in method_defs if stmt.name == "__init__"
        ] + [stmt for stmt in method_defs if stmt.name != "__init__"]

        for stmt in ordered_methods:
            if isinstance(stmt, ast.FunctionDef):
                method_info = self._visit_method_def(stmt, class_name, class_type)
                if method_info is not None:
                    methods[stmt.name] = method_info

        for stmt in node.body:
            if isinstance(stmt, (ast.FunctionDef, ast.Pass)):
                continue
            raise NotImplementedError(
                f"Class body can only contain method definitions, got {type(stmt).__name__}"
            )

        # Capture collected attributes
        attributes = self._pending_attributes
        if attributes:
            field_name_attrs = [
                ir.StringAttr.get(name, self.ctx) for name in attributes.keys()
            ]
            field_type_attrs = [
                ir.TypeAttr.get(attr_type, context=self.ctx)
                for attr_type in attributes.values()
            ]
            class_op.attributes["field_names"] = ir.ArrayAttr.get(
                field_name_attrs, context=self.ctx
            )
            class_op.attributes["field_types"] = ir.ArrayAttr.get(
                field_type_attrs, context=self.ctx
            )
        self._pending_attributes = prev_pending_attrs  # type: ignore
        self._current_class = prev_class
        self._current_class_definition_block = prev_class_definition_block
        self.pop_scope()
        self._set_insertion_block(prev_block)

        # Register the class with methods and attributes
        self.register_class(class_name, class_type, methods, attributes)

    def _visit_method_def(
        self, node: ast.FunctionDef, class_name: str, class_type: ir.Type
    ) -> "MethodInfo | None":
        """
        クラス内のメソッド定義を処理する
        Returns MethodInfo for the method.

        Methods are created at module scope with qualified names (e.g., Counter.__init__)
        to allow FuncObjectOp to reference them using flat symbol references.
        """
        from ._base import MethodInfo

        self._validate_python_function_parameters(
            node, what=f"method '{class_name}.{node.name}'"
        )

        # First argument should be 'self'
        if not node.args.args:
            raise ValueError(f"Method '{node.name}' must have 'self' parameter")

        self_arg = node.args.args[0]
        if self_arg.arg != "self":
            raise ValueError(
                f"First parameter of method '{node.name}' must be 'self', got '{self_arg.arg}'"
            )

        # Build argument types - self uses the class type while explicit
        # primitive annotations stay primitive and Python annotations keep
        # their py.* types.
        entry_arg_types: list[ir.Type] = [class_type]
        arg_type_specs: list[str] = [str(class_type)]
        for arg in node.args.args[1:]:
            arg_type = self.annotation_to_static_class_type(arg.annotation)
            entry_arg_types.append(arg_type)
            arg_type_specs.append(str(arg_type))
        kwonly_arg_types = [
            self.annotation_to_static_class_type(arg.annotation)
            for arg in node.args.kwonlyargs
        ]
        kwonly_type_specs = [str(arg_type) for arg_type in kwonly_arg_types]

        if node.name == "__init__" and node.returns is None:
            result_ir_type = self.get_py_type("!py.none")
        else:
            result_ir_type = self.annotation_to_static_class_type(node.returns)
        result_type_spec = str(result_ir_type)
        if node.name == "__repr__":
            if len(node.args.args) != 1:
                raise ValueError("__repr__ must take only 'self'")
            if result_ir_type != self.get_py_type("!py.str"):
                raise ValueError("__repr__ must return str")
        funcsig = self.build_funcsig(
            arg_type_specs, [result_type_spec], kwonly_types=kwonly_type_specs
        )
        py_func_sig = self.get_py_type(funcsig)

        arg_name_attrs = [
            ir.StringAttr.get(arg.arg, self.ctx) for arg in node.args.args
        ]
        arg_names_attr = (
            ir.ArrayAttr.get(  # pyright: ignore[reportUnknownMemberType]
                arg_name_attrs, context=self.ctx
            )
            if arg_name_attrs
            else None
        )
        kwonly_name_attrs = [
            ir.StringAttr.get(arg.arg, self.ctx) for arg in node.args.kwonlyargs
        ]
        kwonly_names_attr = (
            ir.ArrayAttr.get(kwonly_name_attrs, context=self.ctx)
            if kwonly_name_attrs
            else None
        )

        # Qualified method name at module scope (e.g., Counter.__init__)
        qualified_name = f"{class_name}.{node.name}"

        loc = self._loc(node)
        metadata_prev_block = self.current_block
        metadata_block = getattr(self, "_current_class_definition_block", None)
        if metadata_block is None:
            raise RuntimeError("Missing outer insertion block for method defaults")
        self._set_insertion_block(metadata_block)
        defaults_value, kwdefaults_value = self._build_python_function_metadata(
            node, entry_arg_types[1:], kwonly_arg_types, loc
        )
        self._set_insertion_block(metadata_prev_block)
        positional_default_callable_infos = self.collect_callable_default_infos(
            list(node.args.defaults)
        )
        kwonly_default_callable_infos = self.collect_callable_default_infos(
            list(node.args.kw_defaults)
        )
        # Methods are created at module scope for flat symbol reference
        with loc, ir.InsertionPoint(self.module.body):
            func = py_ops.FuncOp(
                qualified_name,
                ir.TypeAttr.get(py_func_sig),
                arg_names=arg_names_attr,
                nothrow=True,
            )
            if kwonly_names_attr is not None:
                func.attributes["kwonly_names"] = kwonly_names_attr

        with loc:
            all_entry_arg_types = entry_arg_types + kwonly_arg_types
            if all_entry_arg_types:
                entry_block = func.body.blocks.append(*all_entry_arg_types)
            else:
                entry_block = func.body.blocks.append()

        prev_block = self.current_block
        self._set_insertion_block(entry_block)
        self.push_scope()
        self._enter_py_function(qualified_name)
        self.push_function_ast(node)
        self.push_return_type(result_ir_type)
        prev_method = getattr(self, "_current_method", None)
        prev_mutates_self = getattr(self, "_current_method_mutates_self", False)
        self._current_method = node.name
        self._current_method_mutates_self = False

        # Register arguments in scope
        for arg, spec, value in zip(node.args.args, arg_type_specs, entry_block.arguments):
            info = self.maybe_define_callable_parameter_binding(arg.arg, spec, value)
            if info is not None:
                value = self.annotate_known_callable_value(
                    value, info, loc=self._loc(arg)
                )
            self.define_symbol(arg.arg, value)
        offset = len(node.args.args)
        for arg, spec, value in zip(
            node.args.kwonlyargs, kwonly_type_specs, entry_block.arguments[offset:]
        ):
            info = self.maybe_define_callable_parameter_binding(arg.arg, spec, value)
            if info is not None:
                value = self.annotate_known_callable_value(
                    value, info, loc=self._loc(arg)
                )
            self.define_symbol(arg.arg, value)

        # Process method body
        for stmt in node.body:
            self.visit(stmt)

        # Handle implicit return
        active_block = self.current_block or entry_block
        if not self._block_terminated(active_block):
            if result_ir_type != self.get_py_type("!py.none"):
                raise NotImplementedError(
                    f"Method '{node.name}' must explicitly return {result_type_spec}"
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
        mutates_self = bool(getattr(self, "_current_method_mutates_self", False))
        self._current_method = prev_method
        self._current_method_mutates_self = prev_mutates_self
        if node.name == "__init__":
            func.attributes["init_method"] = ir.UnitAttr.get(self.ctx)
        if mutates_self:
            func.attributes["mutates_self"] = ir.UnitAttr.get(self.ctx)

        self.pop_scope()
        self._set_insertion_block(prev_block)

        # Return method info
        return MethodInfo(
            name=node.name,
            arg_types=tuple(entry_arg_types),
            result_types=(result_ir_type,),
            maythrow=maythrow,
            mutates_self=mutates_self,
            init_method=node.name == "__init__",
            arg_names=tuple(arg.arg for arg in node.args.args),
            kwonly_arg_types=tuple(kwonly_arg_types),
            kwonly_names=tuple(arg.arg for arg in node.args.kwonlyargs),
            kwdefault_names=tuple(
                arg.arg
                for arg, default in zip(node.args.kwonlyargs, node.args.kw_defaults)
                if default is not None
            ),
            defaults_count=len(node.args.defaults),
            positional_default_callable_infos=positional_default_callable_infos,
            kwonly_default_callable_infos=kwonly_default_callable_infos,
            defaults=defaults_value,
            kwdefaults=kwdefaults_value,
            returned_function_info=returned_function_info,
            returned_callable_arg_index=returned_callable_arg_index,
        )

    def visit_Return(self, node: ast.Return) -> None:
        """
        関数の返り値を定めるreturn文を処理する

        ```asdl
        Return(expr? value)
        ```
        """
        loc = self._loc(node)
        if node.value is None:
            if self._in_native_func:
                raise ValueError("@native function cannot return None implicitly")
            with loc, self.insertion_point():
                value = py_ops.NoneOp(self.get_py_type("!py.none")).result
        else:
            value = self.require_value(node.value, self.visit(node.value))

        with loc, self.insertion_point():
            expected_return = self.current_return_type()
            if expected_return is not None:
                value = self.coerce_value_to_type(value, expected_return, loc)
                if (
                    node.value is not None
                    and str(expected_return).startswith("!py.func<")
                ):
                    returned_function_info = self.resolve_function_info_from_expression(
                        node.value, value
                    )
                    returned_callable_arg_index = (
                        self.resolve_current_function_parameter_index_from_expression(
                            node.value
                        )
                    )
                    if returned_callable_arg_index is not None:
                        returned_function_info = None
                    self._record_returned_callable_summary(
                        returned_function_info,
                        returned_callable_arg_index,
                    )

            # Use func.return for native functions, py.return for Python functions
            if self._in_native_func:
                func_ops.ReturnOp([value])
            else:
                py_ops.ReturnOp([value])
        self._advance_block_after_terminator()

    def visit_Delete(self, node: ast.Delete) -> None:
        """
        変数を削除するdelete文を処理する

        ```asdl
        Delete(expr* targets)
        ```
        """
        raise NotImplementedError("Delete statement not supported")

    def visit_Assign(self, node: ast.Assign) -> None:
        """
        代入演算子 = を処理する

        ```asdl
        Assign(expr* targets, expr value, string? type_comment)
        ```
        """
        if len(node.targets) != 1:
            raise NotImplementedError("Multiple assignment targets not supported")

        target = node.targets[0]

        # Clear any pending primitive constant from previous assignment
        expr_visitor = self.subvisitors.get("Expr")
        if expr_visitor:
            expr_visitor._pending_prim_const = None

        value = self.require_value(node.value, self.visit(node.value))

        if isinstance(target, ast.Name):
            # Simple assignment: x = value
            if str(value.type).startswith('!py.class<"'):
                allow_fresh_class = (
                    isinstance(node.value, ast.Call)
                    and isinstance(node.value.func, ast.Name)
                    and self.lookup_class(node.value.func.id) is not None
                )
                if not allow_fresh_class:
                    value = self.copy_static_class_value(value, loc=self._loc(node))
            alias_info = self.resolve_function_info_from_expression(node.value, value)
            if alias_info is not None and str(value.type).startswith("!py.func<"):
                value = self.annotate_known_callable_value(
                    value, alias_info, loc=self._loc(node)
                )
            self._check_prim_overwrite(target.id, self._loc(node))
            self.define_symbol(target.id, value)
            if alias_info is not None:
                self.define_function_binding(target.id, alias_info)
            else:
                self.undefine_function_binding(target.id)

            # Check if this was a to_prim() call with a constant value
            # If so, register the constant for cross-region access in @native functions
            if expr_visitor:
                pending = getattr(expr_visitor, "_pending_prim_const", None)
                if pending is not None:
                    mlir_type, const_value = pending
                    self.register_prim_constant(target.id, mlir_type, const_value)
                    expr_visitor._pending_prim_const = None
        elif isinstance(target, ast.Attribute):
            # Attribute assignment: obj.attr = value
            obj = self.require_value(target.value, self.visit(target.value))
            attr_type = self.get_attribute_type(obj.type, target.attr)
            pending_attrs = getattr(self, "_pending_attributes", None)
            if (
                pending_attrs is not None
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
            ):
                current_method = getattr(self, "_current_method", None)
                if current_method is None:
                    raise RuntimeError("Internal error: self assignment outside method")
                if current_method != "__init__" and target.attr not in pending_attrs:
                    raise ValueError(
                        f"Dynamic field introduction outside __init__ is not supported: "
                        f"self.{target.attr}"
                    )
                inferred_type = value.type
                if (
                    target.attr in pending_attrs
                    and pending_attrs[target.attr] != inferred_type
                ):
                    raise TypeError(
                        f"Field '{target.attr}' type mismatch: "
                        f"{pending_attrs[target.attr]} vs {inferred_type}"
                    )
                pending_attrs[target.attr] = inferred_type
                attr_type = inferred_type
                self._current_method_mutates_self = True
            if str(obj.type).startswith('!py.class<"'):
                value = self.coerce_value_to_type(value, attr_type, self._loc(node))
            with self._loc(node), self.insertion_point():
                py_ops.AttrSetOp(obj, target.attr, value)
        else:
            raise NotImplementedError(
                f"Assignment target type {type(target).__name__} not supported"
            )

    def visit_TypeAlias(self, node: ast.TypeAlias) -> None:
        """
        型エイリアスを処理する

        ```asdl
        TypeAlias(
            expr name,
            type_param* type_params,
            expr value
        )
        """
        raise NotImplementedError("Type alias statement not implemented")

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        """
        a += 1 のような累積代入を処理する

        ```asdl
        AugAssign(
            expr target,
            operator op,
            expr value
        )
        """
        rhs = self.require_value(node.value, self.visit(node.value))
        loc = self._loc(node)

        if isinstance(node.target, ast.Name):
            # Simple augmented assignment: x += value
            current = self.lookup_symbol(node.target.id)
            with loc, self.insertion_point():
                result = self._apply_binop(node.op, current, rhs, loc)
            self.define_symbol(node.target.id, result)
            self.undefine_function_binding(node.target.id)

        elif isinstance(node.target, ast.Attribute):
            # Attribute augmented assignment: obj.attr += value
            obj = self.require_value(node.target.value, self.visit(node.target.value))
            attr_type = self.get_attribute_type(obj.type, node.target.attr)
            pending_attrs = getattr(self, "_pending_attributes", None)
            if (
                pending_attrs is not None
                and isinstance(node.target.value, ast.Name)
                and node.target.value.id == "self"
            ):
                if node.target.attr not in pending_attrs:
                    raise ValueError(
                        f"Unknown field '{node.target.attr}' in self mutation"
                    )
                self._current_method_mutates_self = True

            rhs = self.coerce_value_to_type(rhs, attr_type, loc)
            with loc, self.insertion_point():
                current = py_ops.AttrGetOp(attr_type, obj, node.target.attr).result
                result = self._apply_binop(node.op, current, rhs, loc)
                py_ops.AttrSetOp(obj, node.target.attr, result)

        else:
            raise NotImplementedError(
                f"Augmented assignment target type {type(node.target).__name__} not supported"
            )

    def _apply_binop(
        self, op: ast.operator, lhs: ir.Value, rhs: ir.Value, loc: ir.Location
    ) -> ir.Value:
        """二項演算子を適用し、結果を返す"""
        lhs, rhs = self.coerce_operands_for_binary(lhs, rhs, loc)
        if not self.is_py_type(lhs.type) and not self.is_py_type(rhs.type):
            if isinstance(op, ast.Add):
                if self.is_primitive_float_type(lhs.type):
                    return arith_ops.AddFOp(lhs, rhs).result
                if self.is_primitive_int_type(lhs.type):
                    return arith_ops.AddIOp(lhs, rhs).result
            elif isinstance(op, ast.Sub):
                if self.is_primitive_float_type(lhs.type):
                    return arith_ops.SubFOp(lhs, rhs).result
                if self.is_primitive_int_type(lhs.type):
                    return arith_ops.SubIOp(lhs, rhs).result
        if isinstance(op, ast.Add):
            return py_ops.NumAddOp(lhs, rhs).result
        elif isinstance(op, ast.Sub):
            return py_ops.NumSubOp(lhs, rhs).result
        else:
            raise NotImplementedError(f"Operator {type(op).__name__} not supported")

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        """
        c: int のような型注釈を持つ代入を処理する

        ```asdl
        -- 'simple' indicates that we annotate simple name without parens
        AnnAssign(
            expr target,
            expr annotation,
            expr? value,
            int simple
        )
        ```
        """
        raise NotImplementedError(
            "An assignment with a type annotation is not implemented"
        )

    def visit_For(self, node: ast.For) -> None:
        """
        for文の処理をする

        ```asdl
        For(
            expr target,
            expr iter,
            stmt* body,
            stmt* orelse,
            string? type_comment
        )
        ```
        """
        raise NotImplementedError("For statement not implemented")

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        """
        非同期for文の処理をする

        ```asdl
        AsyncFor(
            expr target,
            expr iter,
            stmt* body,
            stmt* orelse,
            string? type_comment
        )
        ```
        """
        raise NotImplementedError("Async for statement not implemented")

    def visit_While(self, node: ast.While) -> None:
        """
        while文を処理する

        ```asdl
        While(
            expr test,
            stmt* body,
            stmt* orelse
        )
        ```
        """
        raise NotImplementedError("While statement not implemented")

    def visit_If(self, node: ast.If) -> None:
        """
        if文:
          if test:
              ...
          else:
              ...

        ```asdl
        If(expr test, stmt* body, stmt* orelse)
        ```
        """
        cond_value = self.require_value(node.test, self.visit(node.test))
        i1 = ir.IntegerType.get_signless(1, context=self.ctx)

        # In native mode, the condition should already be i1 (from arith.cmpi)
        # In object mode, we need to cast from !py.bool to i1
        if self._in_native_func:
            cond = cond_value  # Already i1 from primitive comparison
        else:
            with self._loc(node), self.insertion_point():
                cond = py_ops.CastToPrimOp(
                    i1, cond_value, ir.StringAttr.get("exact", self.ctx)
                ).result

        assert self.current_block is not None
        parent_region = self.current_block.region
        true_block = (
            parent_region.blocks.append()  # pyright: ignore[reportUnknownMemberType]
        )
        false_block = (
            parent_region.blocks.append()  # pyright: ignore[reportUnknownMemberType]
        )
        with self._loc(node), self.insertion_point():
            cf_ops.CondBranchOp(cond, [], [], true_block, false_block)

        def handle_branch(block: ir.Block, statements: list[ast.stmt]) -> bool:
            self._set_insertion_block(block)
            self.push_scope()
            for stmt in statements:
                self.visit(stmt)
            terminated = self._block_terminated(block)
            self.pop_scope()
            return terminated

        true_terminated = handle_branch(true_block, node.body)
        false_terminated = handle_branch(false_block, node.orelse or [])

        if true_terminated and false_terminated:
            self._set_insertion_block(None)
            return

        merge_block = (
            parent_region.blocks.append()  # pyright: ignore[reportUnknownMemberType]
        )
        if not true_terminated:
            with self._loc(node), ir.InsertionPoint(true_block):
                cf_ops.BranchOp([], merge_block)
        if not false_terminated:
            with self._loc(node), ir.InsertionPoint(false_block):
                cf_ops.BranchOp([], merge_block)

        self._set_insertion_block(merge_block)

    def visit_With(self, node: ast.With) -> None:
        """
        with文を処理する

        ```asdl
        With(withitem* items, stmt* body, string? type_comment)
        ```
        """
        raise NotImplementedError("With statement not implemented")

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
        """
        非同期with文を処理する

        ```asdl
        AsyncWith(withitem* items, stmt* body, string? type_comment)
        ```
        """
        raise NotImplementedError("Async with statement not implemented")

    def visit_Match(self, node: ast.Match) -> None:
        """
        match文を処理する

        ```asdl
        Match(expr subject, match_case* cases)
        ```
        """
        raise NotImplementedError("Match statement not implemented")

    def visit_Raise(self, node: ast.Raise) -> None:
        """
        raise文を処理する

        ```asdl
        Raise(expr? exc, expr? cause)
        ```
        """
        if self._in_native_func:
            raise NotImplementedError("raise in @native is not supported")

        with self._loc(node), self.insertion_point():
            self._note_maythrow()
            if node.exc is None:
                py_ops.RaiseCurrentOp()
                self._advance_block_after_terminator()
                return

            if node.cause is not None:
                raise NotImplementedError("raise ... from ... is not supported")

            if isinstance(node.exc, ast.Constant) and isinstance(node.exc.value, str):
                exc_value = self.build_exception_value(
                    message=node.exc.value, loc=self._loc(node)
                )
                py_ops.RaiseOp(exc_value)
                self._advance_block_after_terminator()
                return

            if isinstance(node.exc, ast.Call):
                if (
                    isinstance(node.exc.func, ast.Name)
                    and node.exc.func.id == "Exception"
                ):
                    if len(node.exc.args) != 1 or not isinstance(
                        node.exc.args[0], ast.Constant
                    ):
                        raise NotImplementedError(
                            "Exception(...) requires a single string literal"
                        )
                    if not isinstance(node.exc.args[0].value, str):
                        raise NotImplementedError(
                            "Exception(...) requires a string literal"
                        )
                    exc_value = self.build_exception_value(
                        message=node.exc.args[0].value, loc=self._loc(node)
                    )
                    py_ops.RaiseOp(exc_value)
                    self._advance_block_after_terminator()
                    return

            raise NotImplementedError(
                "raise requires a string literal or bare raise (for now)"
            )

    def visit_Try(self, node: ast.Try) -> None:
        """
        Try文を処理する

        ```asdl
        Try(
            stmt* body,
            excepthandler* handlers,
            stmt* orelse,
            stmt* finalbody
        )
        ```
        """
        raise NotImplementedError("Try statement not implemented")

    def visit_TryStar(self, node: ast.TryStar) -> None:
        """
        except*節が続くtryブロックを処理する

        ```asdl
        TryStar(
            stmt* body,
            excepthandler* handlers,
            stmt* orelse,
            stmt* finalbody
        )
        ```
        """
        raise NotImplementedError("Try star statement not implemented")

    def visit_Assert(self, node: ast.Assert) -> None:
        """
        assert文を処理する

        ```asdl
        Assert(expr test, expr? msg)
        ```
        """
        raise NotImplementedError("Assert statement not implemented")

    def visit_Import(self, node: ast.Import) -> None:
        """
        import文を処理する

        ```asdl
        Import(alias* names)
        ```
        """
        raise NotImplementedError("Import statement not implemented")

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """
        from ... import文を処理する

        ```asdl
        ImportFrom(identifier? module, alias* names, int? level)
        ```

        Handles:
        - from lyrt import native, to_prim, from_prim, alloc, dealloc
        - from lyrt.prim import Int, Float, Vector, Matrix, Tensor
        """
        module = node.module

        if module == "lyrt":
            # Handle lyrt builtins: native, from_prim
            for alias in node.names:
                name = alias.name
                if name in ("native", "to_prim", "from_prim", "alloc", "dealloc"):
                    self._lyrt_builtins.add(name)
                else:
                    raise NotImplementedError(f"Unknown lyrt import: {name}")
            return

        if module == "lyrt.prim":
            # Handle primitive type imports: Int, Float, Vector, Matrix, Tensor
            valid_types = PRIMITIVE_BASE_TYPES | {"Vector", "Matrix", "Tensor"}
            for alias in node.names:
                name = alias.name
                if name in valid_types:
                    # Store the imported name (may be aliased)
                    local_name = alias.asname or name
                    self._prim_types[local_name] = name
                else:
                    raise NotImplementedError(f"Unknown lyrt.prim type: {name}")
            return

        raise NotImplementedError(f"Import from '{module}' not implemented")

    def visit_Global(self, node: ast.Global) -> None:
        """
        global文を処理する

        ```asdl
        Global(identifier* names)
        ```
        """
        raise NotImplementedError("Global statement not implemented")

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        """
        nonlocal文を処理する

        ```asdl
        Nonlocal(identifier* names)
        ```
        """
        raise NotImplementedError("Nonlocal statement not implemented")

    def visit_Expr(self, node: ast.Expr) -> Any:
        """
        式文
        戻り値は破棄してよいので、一応計算はするが特に変数には入れない

        ```asdl
        Expr(expr value)
        ```
        """
        expr_visitor = self.subvisitors.get("Expr")
        if expr_visitor is None:
            raise NotImplementedError("Expression visitor not available")
        expr_visitor.current_block = self.current_block
        expr_visitor.visit(node.value)
        self.current_block = expr_visitor.current_block
        return None

    def visit_Pass(self, node: ast.Pass) -> None:
        """
        pass文を処理する

        ```asdl
        Pass
        ```
        """
        raise NotImplementedError("Pass statement not implemented")

    def visit_Break(self, node: ast.Break) -> None:
        """
        break文を処理する

        ```asdl
        Break
        ```
        """
        raise NotImplementedError("Break statement not implemented")

    def visit_Continue(self, node: ast.Continue) -> None:
        """
        continue文を処理する

        ```asdl
        Continue
        ```
        """
        raise NotImplementedError("Continue statement not implemented")
