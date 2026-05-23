from __future__ import annotations

import ast
from typing import TYPE_CHECKING

from ...mlir import ir
from ...mlir.dialects import _lython_ops_gen as py_ops
from .._base import PRIMITIVE_BASE_TYPES

if TYPE_CHECKING:
    from ..contracts import VisitorRuntime
else:
    VisitorRuntime = object


class ExprCallMixin(VisitorRuntime):
    def visit_Call(self, node: ast.Call) -> ir.Value | None:
        """
        関数呼び出しを処理する。

        ```asdl
        Call(expr func, expr* args, keyword* keywords)
        ```
        """
        loc = self._loc(node)

        # Handle primitive type constructor: Int[32](value), Float[64](value)
        if isinstance(node.func, ast.Subscript):
            if isinstance(node.func.value, ast.Name):
                base_type = node.func.value.id
                # Check if it's an aliased import
                if base_type in self._prim_types:
                    base_type = self._prim_types[base_type]
                if base_type in PRIMITIVE_BASE_TYPES:
                    return self._handle_prim_constructor(node, base_type, loc)
                if base_type in ("Vector", "Matrix", "Tensor"):
                    if len(node.args) != 1:
                        raise ValueError(f"{base_type} constructor requires 1 argument")
                    return self.build_primitive_tensor_constructor(
                        base_type, node.func.slice, node.args[0], loc
                    )

        # Handle Vector/Matrix/Tensor class methods: zeros/ones
        if isinstance(node.func, ast.Attribute) and node.func.attr in ("zeros", "ones"):
            target = node.func.value
            if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name):
                base_type = target.value.id
                if base_type in self._prim_types:
                    base_type = self._prim_types[base_type]
                if base_type in ("Vector", "Matrix", "Tensor"):
                    if node.keywords:
                        raise ValueError(
                            f"{base_type}.{node.func.attr} does not accept keywords"
                        )
                    if node.args:
                        raise ValueError(
                            f"{base_type}.{node.func.attr} does not accept arguments"
                        )
                    fill_value = 0.0 if node.func.attr == "zeros" else 1.0
                    return self.build_primitive_tensor_fill(
                        base_type, target.slice, fill_value, loc
                    )

        # Handle lyrt builtin calls: to_prim/from_prim/alloc/dealloc
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            if func_name == "to_prim" and "to_prim" in self._lyrt_builtins:
                return self._handle_to_prim_call(node, loc)
            if func_name == "from_prim" and "from_prim" in self._lyrt_builtins:
                return self._handle_from_prim(node, loc)
            if func_name == "alloc" and "alloc" in self._lyrt_builtins:
                return self._handle_alloc_call(node, loc)
            if func_name == "dealloc" and "dealloc" in self._lyrt_builtins:
                self._handle_dealloc_call(node, loc)
                return None

        asyncio_builtin = self._resolve_asyncio_builtin(node.func)
        if asyncio_builtin is not None:
            return self._handle_asyncio_builtin_call(asyncio_builtin, node, loc)

        # Check if this is a class instantiation
        if isinstance(node.func, ast.Name):
            class_info = self.lookup_class(node.func.id)
            if class_info is not None:
                return self._handle_class_instantiation(node, class_info, loc)

        # Handle method call early to avoid generating dead py.attr.get
        if isinstance(node.func, ast.Attribute):
            return self._handle_method_call(node, loc)

        # Check if this is a native function call
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            try:
                func_info = self.lookup_function(func_name)
                if func_info.is_async:
                    return self._handle_async_function_call(node, func_info, loc)
                # Check if it's a native function (FunctionType instead of py.func type)
                if isinstance(func_info.func_type, ir.FunctionType):
                    return self._handle_native_call(node, func_info, loc)
            except NameError:
                pass  # Not a registered function, continue with regular handling

        # Otherwise it's a regular function call
        callee = self.require_value(node.func, self.visit(node.func))
        arg_values = [self.require_value(arg, self.visit(arg)) for arg in node.args]
        keyword_arg_nodes = {
            kw.arg: kw.value for kw in node.keywords if kw.arg is not None
        }
        keyword_arg_values = {
            kw.arg: self.require_value(kw.value, self.visit(kw.value))
            for kw in node.keywords
            if kw.arg is not None
        }

        func_info = self.resolve_function_info_from_expression(node.func, callee)
        if func_info is None:
            raise NotImplementedError(
                "Calling function values without recoverable static metadata is not supported yet"
            )
        returned_callable_info = self._resolve_returned_callable_info_from_call(
            returned_function_info=func_info.returned_function_info,
            returned_callable_arg_index=func_info.returned_callable_arg_index,
            positional_param_names=func_info.arg_names,
            kwonly_names=func_info.kwonly_names,
            defaults_count=func_info.defaults_count,
            positional_default_callable_infos=func_info.positional_default_callable_infos,
            kwonly_default_callable_infos=func_info.kwonly_default_callable_infos,
            positional_arg_nodes=list(node.args),
            positional_arg_values=arg_values,
            keyword_arg_nodes=keyword_arg_nodes,
            keyword_arg_values=keyword_arg_values,
            loc=loc,
        )
        result_types = list(func_info.result_types)
        if func_info.maythrow:
            self._note_maythrow()
            if len(result_types) != 1:
                raise NotImplementedError(
                    "py.invoke for multi-result calls is not implemented yet"
                )
        with loc, self.insertion_point():
            if not func_info.has_vararg:
                if node.keywords and self._needs_keyword_callable_materialization(
                    callee
                ):
                    callee = self._build_python_callable(
                        symbol=func_info.symbol,
                        func_type=func_info.func_type,
                        defaults=func_info.defaults,
                        kwdefaults=func_info.kwdefaults,
                        closure=func_info.closure,
                        loc=loc,
                        force_make_function=True,
                    )
                posargs, kwnames, kwvalues = self._build_direct_vectorcall_operands(
                    positional_args=arg_values,
                    keywords=node.keywords,
                    keyword_values=keyword_arg_values,
                    positional_param_types=func_info.arg_types,
                    positional_param_names=func_info.arg_names,
                    kwonly_param_types=func_info.kwonly_arg_types,
                    kwonly_names=func_info.kwonly_names,
                    kwdefault_names=func_info.kwdefault_names,
                    defaults_count=func_info.defaults_count,
                    loc=loc,
                )
            else:
                if node.keywords:
                    raise NotImplementedError(
                        "Keyword arguments for variadic functions are not supported yet"
                    )
                object_args = [
                    self.ensure_object(value, loc=loc) for value in arg_values
                ]
                posargs = self.build_tuple(object_args, loc=loc)
                empty_tuple_type = self.get_py_type("!py.tuple<>")
                kwnames = py_ops.TupleEmptyOp(empty_tuple_type).result
                kwvalues = py_ops.TupleEmptyOp(empty_tuple_type).result

        if func_info.maythrow:
            if result_types == [self.get_py_type("!py.none")]:
                normal_block = self._emit_none_returning_invoke(
                    callee,
                    posargs,
                    kwnames,
                    kwvalues,
                    loc,
                )
                self._set_insertion_block(normal_block)
                with loc, self.insertion_point():
                    none_val = py_ops.NoneOp(self.get_py_type("!py.none")).result
                return none_val
            return self._emit_value_returning_invoke(
                callee,
                posargs,
                kwnames,
                kwvalues,
                result_types[0],
                loc,
                returned_function_info=returned_callable_info,
            )

        if len(result_types) != 1:
            raise NotImplementedError("Only single-result functions supported")
        with loc, self.insertion_point():
            call = py_ops.CallVectorOp(result_types, callee, posargs, kwnames, kwvalues)
            result = call.results_[0]
            if str(result.type).startswith("!py.func<"):
                return self._materialize_known_callable_result(
                    result, returned_callable_info, loc
                )
            self._attach_returned_callable_info(call.operation, returned_callable_info)
            return result
