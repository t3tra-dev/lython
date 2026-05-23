from __future__ import annotations

import ast
from typing import TYPE_CHECKING

from ...frontend.symbols import ClassInfo
from ...mlir import ir
from ...mlir.dialects import _lython_ops_gen as py_ops

if TYPE_CHECKING:
    from ..contracts import VisitorRuntime
else:
    VisitorRuntime = object


class ExprCallMethodsMixin(VisitorRuntime):
    def _handle_class_instantiation(
        self, node: ast.Call, class_info: ClassInfo, loc: ir.Location
    ) -> ir.Value:
        """Handle class instantiation: ClassName(args)"""

        arg_values = [self.require_value(arg, self.visit(arg)) for arg in node.args]

        with loc, self.insertion_point():
            # Create new instance
            instance = py_ops.ClassNewOp(class_info.class_type, class_info.name).result

            # Call __init__ if it exists
            if "__init__" in class_info.methods:
                init_info = class_info.methods["__init__"]
                if init_info.maythrow:
                    self._note_maythrow()
                    if list(init_info.result_types) != [self.get_py_type("!py.none")]:
                        raise NotImplementedError(
                            "py.invoke for value-returning __init__ is not implemented yet"
                        )
                posargs, kwnames, kwvalues = self._build_direct_vectorcall_operands(
                    positional_args=arg_values,
                    keywords=node.keywords,
                    positional_param_types=init_info.arg_types[1:],
                    positional_param_names=init_info.arg_names[1:],
                    kwonly_param_types=init_info.kwonly_arg_types,
                    kwonly_names=init_info.kwonly_names,
                    kwdefault_names=init_info.kwdefault_names,
                    defaults_count=init_info.defaults_count,
                    loc=loc,
                    leading_args=[instance],
                )

                # Get __init__ function reference
                init_funcsig = self.build_funcsig(
                    [str(t) for t in init_info.arg_types],
                    [str(t) for t in init_info.result_types],
                    kwonly_types=[str(t) for t in init_info.kwonly_arg_types],
                )
                init_func_type = self.get_py_type(f"!py.func<{init_funcsig}>")
                init_func = self._build_method_callable(
                    symbol=f"{class_info.name}.__init__",
                    func_type=init_func_type,
                    defaults=init_info.defaults,
                    kwdefaults=init_info.kwdefaults,
                    loc=loc,
                    force_make_function=bool(node.keywords),
                )

                if init_info.maythrow:
                    normal_block = self._emit_none_returning_invoke(
                        init_func, posargs, kwnames, kwvalues, loc
                    )
                    self._set_insertion_block(normal_block)
                else:
                    # Call __init__ (result is None, we ignore it)
                    py_ops.CallVectorOp(
                        list(init_info.result_types),
                        init_func,
                        posargs,
                        kwnames,
                        kwvalues,
                    )

            return instance

    def _handle_method_call(self, node: ast.Call, loc: ir.Location) -> ir.Value:
        """Handle method call: obj.method(args)"""
        assert isinstance(node.func, ast.Attribute)

        obj = self.require_value(node.func.value, self.visit(node.func.value))
        method_name = node.func.attr
        arg_values = [self.require_value(arg, self.visit(arg)) for arg in node.args]
        keyword_arg_nodes = {
            kw.arg: kw.value for kw in node.keywords if kw.arg is not None
        }
        keyword_arg_values = {
            kw.arg: self.require_value(kw.value, self.visit(kw.value))
            for kw in node.keywords
            if kw.arg is not None
        }

        list_elem_type = self.get_list_element_type(obj.type)
        if list_elem_type is not None:
            if method_name not in {"append", "remove"}:
                raise NotImplementedError(
                    f"List method '{method_name}' is not supported"
                )
            if node.keywords:
                raise NotImplementedError(
                    f"Keyword arguments for list.{method_name}() are not supported yet"
                )
            if len(arg_values) != 1:
                raise ValueError(f"list.{method_name}() expects exactly 1 argument")

            value = self.coerce_value_to_type(arg_values[0], list_elem_type, loc)
            value = self._prepare_list_element_for_storage(value, list_elem_type, loc)

            stmt_visitor = self.subvisitors.get("Stmt")
            if (
                stmt_visitor is not None
                and isinstance(node.func.value, ast.Attribute)
                and isinstance(node.func.value.value, ast.Name)
                and node.func.value.value.id == "self"
            ):
                stmt_visitor._current_method_mutates_self = True

            with loc, self.insertion_point():
                if method_name == "append":
                    py_ops.ListAppendOp(obj, value)
                else:
                    py_ops.ListRemoveOp(obj, value)
                return py_ops.NoneOp(self.get_py_type("!py.none")).result

        obj_type_str = str(obj.type)
        if obj_type_str.startswith("!py.task<"):
            if method_name != "cancel":
                raise NotImplementedError(
                    f"Task method '{method_name}' is not supported"
                )
            if node.keywords:
                raise NotImplementedError(
                    "Keyword arguments for task.cancel() are not supported"
                )
            if arg_values:
                raise ValueError("task.cancel() expects no arguments")
            with loc, self.insertion_point():
                return py_ops.TaskCancelOp(self.get_py_type("!py.bool"), obj).accepted

        # For now, we need to determine the class from the object type
        # This is a simplified implementation

        # Extract class name from type like !py.class<"Counter">
        if obj_type_str.startswith('!py.class<"') and obj_type_str.endswith('">'):
            class_name = obj_type_str[len('!py.class<"') : -len('">')]  # noqa
        else:
            raise NotImplementedError(
                f"Method calls only supported on class instances, got {obj_type_str}"
            )

        class_info = self.lookup_class(class_name)
        if class_info is None:
            raise NameError(f"Unresolved class '{class_name}'")

        if method_name not in class_info.methods:
            raise AttributeError(f"Class '{class_name}' has no method '{method_name}'")

        method_info = class_info.methods[method_name]
        returned_callable_info = self._resolve_returned_callable_info_from_call(
            returned_function_info=method_info.returned_function_info,
            returned_callable_arg_index=method_info.returned_callable_arg_index,
            positional_param_names=method_info.arg_names,
            kwonly_names=method_info.kwonly_names,
            defaults_count=method_info.defaults_count,
            positional_default_callable_infos=method_info.positional_default_callable_infos,
            kwonly_default_callable_infos=method_info.kwonly_default_callable_infos,
            positional_arg_nodes=[node.func.value, *list(node.args)],
            positional_arg_values=[obj, *arg_values],
            keyword_arg_nodes=keyword_arg_nodes,
            keyword_arg_values=keyword_arg_values,
            loc=loc,
        )
        if method_info.maythrow:
            self._note_maythrow()
            if len(method_info.result_types) != 1:
                raise NotImplementedError(
                    "py.invoke for multi-result method calls is not implemented yet"
                )

        with loc, self.insertion_point():
            posargs, kwnames, kwvalues = self._build_direct_vectorcall_operands(
                positional_args=arg_values,
                keywords=node.keywords,
                keyword_values=keyword_arg_values,
                positional_param_types=method_info.arg_types[1:],
                positional_param_names=method_info.arg_names[1:],
                kwonly_param_types=method_info.kwonly_arg_types,
                kwonly_names=method_info.kwonly_names,
                kwdefault_names=method_info.kwdefault_names,
                defaults_count=method_info.defaults_count,
                loc=loc,
                leading_args=[obj],
            )

            # Get method function reference
            method_funcsig = self.build_funcsig(
                [str(t) for t in method_info.arg_types],
                [str(t) for t in method_info.result_types],
                kwonly_types=[str(t) for t in method_info.kwonly_arg_types],
            )
            method_func_type = self.get_py_type(f"!py.func<{method_funcsig}>")
            method_func = self._build_method_callable(
                symbol=f"{class_name}.{method_name}",
                func_type=method_func_type,
                defaults=method_info.defaults,
                kwdefaults=method_info.kwdefaults,
                loc=loc,
                force_make_function=bool(node.keywords),
            )

            result_types = list(method_info.result_types)
            if method_info.maythrow:
                if result_types == [self.get_py_type("!py.none")]:
                    normal_block = self._emit_none_returning_invoke(
                        method_func, posargs, kwnames, kwvalues, loc
                    )
                    self._set_insertion_block(normal_block)
                    with loc, self.insertion_point():
                        return py_ops.NoneOp(self.get_py_type("!py.none")).result
                return self._emit_value_returning_invoke(
                    method_func,
                    posargs,
                    kwnames,
                    kwvalues,
                    result_types[0],
                    loc,
                    returned_function_info=returned_callable_info,
                )

            # Call the method
            call = py_ops.CallVectorOp(
                result_types, method_func, posargs, kwnames, kwvalues
            )
            result = call.results_[0]
            if str(result.type).startswith("!py.func<"):
                return self._materialize_known_callable_result(
                    result, returned_callable_info, loc
                )
            self._attach_returned_callable_info(call.operation, returned_callable_info)
            return result
