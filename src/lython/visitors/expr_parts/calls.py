# pyright: reportAttributeAccessIssue=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false
from __future__ import annotations

import ast

from ...mlir import ir
from ...mlir.dialects import _async_ops_gen as async_ops
from ...mlir.dialects import _lython_ops_gen as py_ops
from ...mlir.dialects import func as func_ops
from .._base import PRIMITIVE_BASE_TYPES, FunctionInfo


class ExprCallMixin:
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

        normal_block = None

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

            if func_info.maythrow and result_types == [self.get_py_type("!py.none")]:
                parent_region = self.current_block.region  # type: ignore[union-attr]
                normal_block = parent_region.blocks.append()  # type: ignore[reportUnknownMemberType]
                unwind_block = parent_region.blocks.append(  # type: ignore[reportUnknownMemberType]
                    self.get_py_type("!py.exception")
                )
                exc_null = py_ops.ExceptionNullOp(
                    self.get_py_type("!py.exception")
                ).result
                py_ops.InvokeOp(
                    callee,
                    posargs,
                    kwnames,
                    kwvalues,
                    [],
                    [exc_null],
                    normal_block,
                    unwind_block,
                )
                with ir.InsertionPoint(unwind_block), loc:
                    py_ops.RaiseCurrentOp()

        if func_info.maythrow:
            if result_types == [self.get_py_type("!py.none")]:
                if normal_block is None:
                    raise RuntimeError(
                        "Internal error: normal_block was not created for maythrow call"
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

    def _handle_async_function_call(
        self, node: ast.Call, func_info: FunctionInfo, loc: ir.Location
    ) -> ir.Value:
        coerced_args = self._build_async_function_call_args(node, func_info, loc)
        if len(func_info.result_types) != 1:
            raise NotImplementedError("Async functions must have one payload result")
        result_type = self.get_py_type(f"!py.coro<{func_info.result_types[0]}>")
        with loc, self.insertion_point():
            return py_ops.CoroCreateOp(
                result_type,
                ir.FlatSymbolRefAttr.get(func_info.symbol, self.ctx),
                coerced_args,
            ).result

    def _resolve_direct_async_call(
        self, node: ast.expr
    ) -> tuple[ast.Call, FunctionInfo] | None:
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            return None
        try:
            func_info = self.lookup_function(node.func.id)
        except NameError:
            return None
        if not func_info.is_async:
            return None
        return node, func_info

    def _build_async_function_call_args(
        self, node: ast.Call, func_info: FunctionInfo, loc: ir.Location
    ) -> list[ir.Value]:
        if node.keywords:
            raise NotImplementedError(
                "Keyword arguments for async functions are not supported yet"
            )
        if len(node.args) != len(func_info.arg_types):
            raise ValueError(
                f"Async function '{func_info.symbol}' expects {len(func_info.arg_types)} "
                f"arguments, got {len(node.args)}"
            )
        arg_values = [self.require_value(arg, self.visit(arg)) for arg in node.args]
        return [
            self.coerce_value_to_type(arg, expected, loc)
            for arg, expected in zip(arg_values, func_info.arg_types)
        ]

    def _emit_direct_async_call(
        self, node: ast.Call, func_info: FunctionInfo, loc: ir.Location
    ) -> ir.Value:
        coerced_args = self._build_async_function_call_args(node, func_info, loc)
        if len(func_info.result_types) != 1:
            raise NotImplementedError("Async functions must have one payload result")
        payload_type = func_info.result_types[0]
        async_value_type = ir.Type.parse(f"!async.value<{payload_type}>", self.ctx)
        with loc, self.insertion_point():
            return async_ops.CallOp(
                [async_value_type],
                ir.FlatSymbolRefAttr.get(func_info.symbol, self.ctx),
                coerced_args,
            ).operation.results[0]

    def _emit_immediate_async_call_await(
        self, node: ast.expr, loc: ir.Location
    ) -> ir.Value | None:
        resolved = self._resolve_direct_async_call(node)
        if resolved is None:
            return None
        call_node, func_info = resolved
        awaitable = self._handle_async_function_call(call_node, func_info, loc)
        payload_type = self.get_awaitable_payload_type(awaitable.type)
        if payload_type is None:
            raise TypeError(
                f"Internal error: async call did not produce an awaitable, got {awaitable.type}"
            )
        with loc, self.insertion_point():
            return py_ops.AwaitOp(payload_type, awaitable).result

    def _resolve_asyncio_call(self, node: ast.expr, name: str) -> ast.Call | None:
        if not isinstance(node, ast.Call):
            return None
        if self._resolve_asyncio_builtin(node.func) != name:
            return None
        return node

    def _emit_asyncio_gather(self, node: ast.Call, loc: ir.Location) -> ir.Value:
        if node.keywords:
            raise NotImplementedError(
                "asyncio.gather keyword arguments are unsupported"
            )
        awaitables: list[ir.Value] = []
        payload_types: list[ir.Type] = []
        for arg in node.args:
            resolved = self._resolve_direct_async_call(arg)
            if resolved is not None:
                call_node, func_info = resolved
                awaitable = self._handle_async_function_call(call_node, func_info, loc)
            else:
                awaitable = self.require_value(arg, self.visit(arg))
            payload_type = self.get_awaitable_payload_type(awaitable.type)
            if payload_type is None:
                raise TypeError(
                    "asyncio.gather expects statically typed awaitables, "
                    f"got {awaitable.type}"
                )
            awaitables.append(awaitable)
            payload_types.append(payload_type)

        if not awaitables:
            return self.build_tuple([], loc=loc)

        tuple_spec = ", ".join(str(payload_type) for payload_type in payload_types)
        tuple_type = self.get_py_type(f"!py.tuple<{tuple_spec}>")
        with loc, self.insertion_point():
            return py_ops.AsyncGatherOp(tuple_type, awaitables).result

    def _resolve_asyncio_builtin(self, func: ast.expr) -> str | None:
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            if self._static_modules.get(func.value.id) == "asyncio":
                if func.attr in {"run", "create_task", "gather", "sleep"}:
                    return func.attr
                raise NotImplementedError(f"asyncio.{func.attr} is not supported yet")
        if isinstance(func, ast.Name):
            binding = self._static_module_symbols.get(func.id)
            if binding is not None:
                module, name = binding
                if module == "asyncio":
                    return name
        return None

    def _handle_asyncio_builtin_call(
        self, name: str, node: ast.Call, loc: ir.Location
    ) -> ir.Value:
        if node.keywords:
            raise NotImplementedError(
                f"asyncio.{name} keyword arguments are unsupported"
            )

        if name == "run":
            if self.in_async_function():
                raise NotImplementedError(
                    "asyncio.run cannot be called from an async function"
                )
            if len(node.args) != 1:
                raise ValueError("asyncio.run expects exactly one awaitable")
            gather_call = self._resolve_asyncio_call(node.args[0], "gather")
            if gather_call is not None:
                return self._emit_asyncio_gather(gather_call, loc)
            immediate = self._emit_immediate_async_call_await(node.args[0], loc)
            if immediate is not None:
                return immediate
            awaitable = self.require_value(node.args[0], self.visit(node.args[0]))
            payload_type = self.get_awaitable_payload_type(awaitable.type)
            if payload_type is None:
                raise TypeError(
                    f"asyncio.run expects a statically typed awaitable, got {awaitable.type}"
                )
            with loc, self.insertion_point():
                return py_ops.AwaitOp(payload_type, awaitable).result

        if name == "create_task":
            if len(node.args) != 1:
                raise ValueError("asyncio.create_task expects exactly one coroutine")
            coroutine = self.require_value(node.args[0], self.visit(node.args[0]))
            payload_type = self.get_awaitable_payload_type(coroutine.type)
            if payload_type is None or not str(coroutine.type).startswith("!py.coro<"):
                raise TypeError(
                    "asyncio.create_task expects a statically typed coroutine, "
                    f"got {coroutine.type}"
                )
            task_type = self.get_py_type(f"!py.task<{payload_type}>")
            with loc, self.insertion_point():
                return py_ops.TaskCreateOp(task_type, coroutine).result

        if name == "gather":
            raise NotImplementedError(
                "asyncio.gather must be immediately awaited or passed to asyncio.run"
            )

        if name == "sleep":
            if len(node.args) != 1:
                raise ValueError("asyncio.sleep expects exactly one duration")
            seconds = self.require_value(node.args[0], self.visit(node.args[0]))
            future_type = self.get_py_type("!py.future<!py.none>")
            with loc, self.insertion_point():
                return py_ops.AsyncSleepOp(future_type, seconds).result

        raise NotImplementedError(f"asyncio.{name} is not supported yet")

    def _handle_native_call(
        self, node: ast.Call, func_info: FunctionInfo, loc: ir.Location
    ) -> ir.Value:
        """Handle call to a @native function using func.call."""

        arg_values = [self.require_value(arg, self.visit(arg)) for arg in node.args]

        if len(arg_values) != len(func_info.arg_types):
            raise ValueError(
                f"Native function '{func_info.symbol}' expects {len(func_info.arg_types)} "
                f"arguments, got {len(arg_values)}"
            )
        coerced_args = [
            self.coerce_value_to_type(arg, expected, loc)
            for arg, expected in zip(arg_values, func_info.arg_types)
        ]

        result_types = list(func_info.result_types)

        with loc, self.insertion_point():
            call = func_ops.CallOp(result_types, func_info.symbol, coerced_args)
            return call.results[0]

    def _split_type_specs(self, specs: str) -> list[str]:
        parts: list[str] = []
        depth = 0
        current: list[str] = []
        for ch in specs:
            if ch == "<":
                depth += 1
            elif ch == ">":
                depth -= 1
            elif ch == "," and depth == 0:
                part = "".join(current).strip()
                if part:
                    parts.append(part)
                current = []
                continue
            current.append(ch)
        part = "".join(current).strip()
        if part:
            parts.append(part)
        return parts

    def _build_invoke_result_seed(
        self, result_type: ir.Type, loc: ir.Location
    ) -> ir.Value:
        type_str = str(result_type)

        if type_str == "!py.none":
            with loc, self.insertion_point():
                return py_ops.NoneOp(result_type).result
        if type_str == "!py.exception":
            with loc, self.insertion_point():
                return py_ops.ExceptionNullOp(result_type).result
        if type_str == "!py.traceback":
            with loc, self.insertion_point():
                return py_ops.TracebackNullOp(result_type).result
        if type_str == "!py.location":
            with loc, self.insertion_point():
                return py_ops.LocationCurrentOp(result_type).result
        if type_str == "!py.object":
            with loc, self.insertion_point():
                none_val = py_ops.NoneOp(self.get_py_type("!py.none")).result
            return self.ensure_object(none_val, loc=loc)
        if type_str in {"!py.int", "!py.bool"}:
            with loc, self.insertion_point():
                return py_ops.IntConstantOp(result_type, "0").result
        if type_str == "!py.float":
            with loc, self.insertion_point():
                return py_ops.FloatConstantOp(result_type, 0.0).result
        if type_str == "!py.str":
            with loc, self.insertion_point():
                return py_ops.StrConstantOp(
                    result_type, ir.StringAttr.get("", self.ctx)
                ).result
        if type_str == "!py.tuple<>":
            with loc, self.insertion_point():
                return py_ops.TupleEmptyOp(result_type).result
        if type_str.startswith("!py.tuple<") and type_str.endswith(">"):
            inner = type_str[len("!py.tuple<") : -1].strip()
            if not inner:
                with loc, self.insertion_point():
                    return py_ops.TupleEmptyOp(result_type).result
            elements = [
                self._build_invoke_result_seed(self.get_py_type(spec), loc)
                for spec in self._split_type_specs(inner)
            ]
            return self.build_tuple(elements, loc=loc)
        if type_str.startswith("!py.dict<"):
            with loc, self.insertion_point():
                return py_ops.DictEmptyOp(result_type).result
        if type_str.startswith("!py.list<"):
            with loc, self.insertion_point():
                return py_ops.ListNewOp(result_type).result
        if type_str.startswith('!py.class<"') and type_str.endswith('">'):
            class_name = type_str[len('!py.class<"') : -len('">')]
            with loc, self.insertion_point():
                return py_ops.ClassNewOp(result_type, class_name).result
        if type_str.startswith("i"):
            return self._build_primitive_scalar(0, result_type, loc)
        if type_str.startswith("f"):
            return self._build_primitive_scalar(0.0, result_type, loc)

        raise NotImplementedError(
            f"py.invoke result placeholder for {result_type} is not implemented yet"
        )

    def _emit_none_returning_invoke(
        self,
        callee: ir.Value,
        posargs: ir.Value,
        kwnames: ir.Value,
        kwvalues: ir.Value,
        loc: ir.Location,
    ) -> ir.Block:
        with loc:
            parent_region = self.current_block.region  # type: ignore[union-attr]
            normal_block = parent_region.blocks.append()  # type: ignore[reportUnknownMemberType]
            unwind_block = parent_region.blocks.append(  # type: ignore[reportUnknownMemberType]
                self.get_py_type("!py.exception")
            )
        with loc, self.insertion_point():
            exc_null = py_ops.ExceptionNullOp(self.get_py_type("!py.exception")).result
            py_ops.InvokeOp(
                callee,
                posargs,
                kwnames,
                kwvalues,
                [],
                [exc_null],
                normal_block,
                unwind_block,
            )
        with ir.InsertionPoint(unwind_block), loc:
            py_ops.RaiseCurrentOp()
        return normal_block

    def _emit_value_returning_invoke(
        self,
        callee: ir.Value,
        posargs: ir.Value,
        kwnames: ir.Value,
        kwvalues: ir.Value,
        result_type: ir.Type,
        loc: ir.Location,
        returned_function_info: FunctionInfo | None = None,
    ) -> ir.Value:
        with loc:
            parent_region = self.current_block.region  # type: ignore[union-attr]
            normal_block = parent_region.blocks.append(  # type: ignore[reportUnknownMemberType]
                result_type
            )
            unwind_block = parent_region.blocks.append(  # type: ignore[reportUnknownMemberType]
                self.get_py_type("!py.exception")
            )
        with loc, self.insertion_point():
            if returned_function_info is not None and str(result_type).startswith(
                "!py.func<"
            ):
                seed = py_ops.FuncObjectOp(
                    result_type,
                    ir.FlatSymbolRefAttr.get(
                        returned_function_info.symbol,
                        self.ctx,
                    ),
                ).result
            else:
                seed = self._build_invoke_result_seed(result_type, loc)
            exc_null = py_ops.ExceptionNullOp(self.get_py_type("!py.exception")).result
            py_ops.InvokeOp(
                callee,
                posargs,
                kwnames,
                kwvalues,
                [seed],
                [exc_null],
                normal_block,
                unwind_block,
            )
        with ir.InsertionPoint(unwind_block), loc:
            py_ops.RaiseCurrentOp()
        self._set_insertion_block(normal_block)
        result = normal_block.arguments[0]
        if returned_function_info is not None and str(result_type).startswith(
            "!py.func<"
        ):
            result = self._materialize_known_callable_result(
                result, returned_function_info, loc
            )
        return result
