# pyright: reportAttributeAccessIssue=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false
from __future__ import annotations

import ast

from ...mlir import ir
from ...mlir.dialects import _lython_ops_gen as py_ops
from .._base import MethodInfo


class StmtClassMixin:
    """Statement lowering for class and method definitions."""

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        if node.bases:
            raise NotImplementedError("Class inheritance not yet supported")
        if node.decorator_list:
            raise NotImplementedError("Class decorators not yet supported")

        class_name = node.name
        loc = self._loc(node)

        with loc, ir.InsertionPoint(self.module.body):
            class_op = py_ops.ClassOp(class_name)

        class_body_block = class_op.body.blocks.append()
        class_type = self.get_py_type(f'!py.class<"{class_name}">')

        prev_block = self.current_block
        self._set_insertion_block(class_body_block)
        self.push_scope()

        prev_class = getattr(self, "_current_class", None)
        self._current_class = class_name
        prev_class_definition_block = getattr(
            self, "_current_class_definition_block", None
        )
        self._current_class_definition_block = prev_block

        prev_pending_attrs = getattr(self, "_pending_attributes", None)
        self._pending_attributes: dict[str, ir.Type] = {}

        methods: dict[str, MethodInfo] = {}
        method_defs = [stmt for stmt in node.body if isinstance(stmt, ast.FunctionDef)]
        ordered_methods = [stmt for stmt in method_defs if stmt.name == "__init__"] + [
            stmt for stmt in method_defs if stmt.name != "__init__"
        ]

        for stmt in ordered_methods:
            method_info = self._visit_method_def(stmt, class_name, class_type)
            if method_info is not None:
                methods[stmt.name] = method_info

        for stmt in node.body:
            if isinstance(stmt, (ast.FunctionDef, ast.Pass)):
                continue
            raise NotImplementedError(
                f"Class body can only contain method definitions, got {type(stmt).__name__}"
            )

        attributes = self._pending_attributes
        if attributes:
            field_name_attrs = [
                ir.StringAttr.get(name, self.ctx) for name in attributes.keys()
            ]
            field_type_attrs = [
                ir.TypeAttr.get(attr_type, context=self.ctx)
                for attr_type in attributes.values()
            ]
            class_op.attributes["field_names"] = self.array_attr(field_name_attrs)
            class_op.attributes["field_types"] = self.array_attr(field_type_attrs)
        self._pending_attributes = prev_pending_attrs  # type: ignore
        self._current_class = prev_class
        self._current_class_definition_block = prev_class_definition_block
        self.pop_scope()
        self._set_insertion_block(prev_block)

        self.register_class(class_name, class_type, methods, attributes)

    def _visit_method_def(
        self, node: ast.FunctionDef, class_name: str, class_type: ir.Type
    ) -> MethodInfo | None:
        self._validate_python_function_parameters(
            node, what=f"method '{class_name}.{node.name}'"
        )

        if not node.args.args:
            raise ValueError(f"Method '{node.name}' must have 'self' parameter")

        self_arg = node.args.args[0]
        if self_arg.arg != "self":
            raise ValueError(
                f"First parameter of method '{node.name}' must be 'self', got '{self_arg.arg}'"
            )

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
        funcsig = self.build_funcsig(
            arg_type_specs, [result_type_spec], kwonly_types=kwonly_type_specs
        )
        py_func_sig = self.get_py_type(funcsig)

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

        for stmt in node.body:
            self.visit(stmt)

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
