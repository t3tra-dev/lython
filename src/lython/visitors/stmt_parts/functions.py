from __future__ import annotations

import ast

from ...mlir import ir
from ...mlir.dialects import _lython_ops_gen as py_ops
from .async_functions import AsyncFunctionMixin
from .function_support import FunctionSupportMixin
from .native_functions import NativeFunctionMixin
from .nested_functions import NestedFunctionMixin


class StmtFunctionMixin(
    FunctionSupportMixin,
    NestedFunctionMixin,
    NativeFunctionMixin,
    AsyncFunctionMixin,
):
    """Statement lowering entry points for Python function definitions."""

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if self._in_native_func:
            raise NotImplementedError(
                "Nested Python functions inside @native are not supported"
            )
        if self.is_nested_function_context():
            return self._visit_nested_function_def(node)

        native_info = self._get_native_decorator(node.decorator_list)
        if native_info is not None:
            return self._visit_native_function_def(node, native_info)

        self._validate_python_function_parameters(node, what=f"function '{node.name}'")

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
        for arg, spec, value in zip(
            node.args.args, arg_type_specs, entry_block.arguments
        ):
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
