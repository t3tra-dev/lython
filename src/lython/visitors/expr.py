from __future__ import annotations

import ast

from ..mlir import ir
from ..mlir.dialects import _lython_ops_gen as py_ops
from ..mlir.dialects import arith as arith_ops
from ..mlir.dialects import func as func_ops
from ..mlir.dialects import tensor as tensor_ops
from ._base import PRIMITIVE_BASE_TYPES, BaseVisitor, ClassInfo, FunctionInfo

__all__ = ["ExprVisitor"]


class ExprVisitor(BaseVisitor):
    """
    式(expr)ノードの訪問を担当するクラス
    以下のようなオブジェクト型に対応する
    - int => PyInt (PyObject派生)
    - bool => PyBool (PyObject派生)
    - str => PyUnicodeObject (PyObject派生)
    - list => PyListObject (PyObject派生)
    - dict => PyDictObject (PyObject派生)

    ```asdl
          -- BoolOp() can use left & right?
    expr = BoolOp(boolop op, expr* values)
         | NamedExpr(expr target, expr value)
         | BinOp(expr left, operator op, expr right)
         | UnaryOp(unaryop op, expr operand)
         | Lambda(arguments args, expr body)
         | IfExp(expr test, expr body, expr orelse)
         | Dict(expr* keys, expr* values)
         | Set(expr* elts)
         | ListComp(expr elt, comprehension* generators)
         | SetComp(expr elt, comprehension* generators)
         | DictComp(expr key, expr value, comprehension* generators)
         | GeneratorExp(expr elt, comprehension* generators)
         -- the grammar constrains where yield expressions can occur
         | Await(expr value)
         | Yield(expr? value)
         | YieldFrom(expr value)
         -- need sequences for compare to distinguish between
         -- x < 4 < 3 and (x < 4) < 3
         | Compare(expr left, cmpop* ops, expr* comparators)
         | Call(expr func, expr* args, keyword* keywords)
         | FormattedValue(expr value, int conversion, expr? format_spec)
         | JoinedStr(expr* values)
         | Constant(constant value, string? kind)

         -- the following expression can appear in assignment context
         | Attribute(expr value, identifier attr, expr_context ctx)
         | Subscript(expr value, expr slice, expr_context ctx)
         | Starred(expr value, expr_context ctx)
         | Name(identifier id, expr_context ctx)
         | List(expr* elts, expr_context ctx)
         | Tuple(expr* elts, expr_context ctx)

         -- can appear only in Subscript
         | Slice(expr? lower, expr? upper, expr? step)

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

    def visit_BoolOp(self, node: ast.BoolOp) -> None:
        """
        ブールの論理演算を処理する

        ```asdl
        BoolOp(boolop op, expr* values)
        ```
        """
        raise NotImplementedError("Boolean operations not implemented")

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        """
        名前付き式を処理する

        ```asdl
        NamedExpr(expr target, expr value)
        ```
        """
        raise NotImplementedError("Named expression not implemented")

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
    ) -> tuple[ir.Value, ir.Value, ir.Value]:
        positional_param_types = list(positional_param_types)
        positional_param_names = list(positional_param_names)
        kwonly_param_types = list(kwonly_param_types)
        kwonly_names = list(kwonly_names)
        kwdefault_names = list(kwdefault_names)
        leading_args = list(leading_args or [])

        if len(positional_args) > len(positional_param_types):
            raise ValueError(
                f"Expected at most {len(positional_param_types)} positional "
                f"arguments, got {len(positional_args)}"
            )

        coerced_positional = [
            self.coerce_value_to_type(arg, expected, loc)
            for arg, expected in zip(positional_args, positional_param_types)
        ]
        posargs = self.build_tuple(leading_args + coerced_positional, loc=loc)

        empty_tuple_type = self.get_py_type("!py.tuple<>")
        if not keywords:
            min_positional_count = len(positional_param_types) - defaults_count
            if len(positional_args) < min_positional_count:
                raise ValueError(
                    f"Expected at least {min_positional_count} positional "
                    f"arguments, got {len(positional_args)}"
                )
            with loc, self.insertion_point():
                return (
                    posargs,
                    py_ops.TupleEmptyOp(empty_tuple_type).result,
                    py_ops.TupleEmptyOp(empty_tuple_type).result,
                )

        positional_types = {
            name: (index, ty)
            for index, (name, ty) in enumerate(
                zip(positional_param_names, positional_param_types)
            )
        }
        kwonly_types = {
            name: ty for name, ty in zip(kwonly_names, kwonly_param_types)
        }
        kwdefault_name_set = set(kwdefault_names)
        seen_keywords: set[str] = set()
        kwname_values: list[ir.Value] = []
        kwvalue_values: list[ir.Value] = []

        with loc, self.insertion_point():
            for keyword in keywords:
                if keyword.arg is None:
                    raise NotImplementedError(
                        "Keyword argument unpacking is not supported yet"
                    )
                if keyword.arg in seen_keywords:
                    raise ValueError(f"Duplicate keyword argument '{keyword.arg}'")
                seen_keywords.add(keyword.arg)
                if keyword.arg in positional_types:
                    slot_index, expected_type = positional_types[keyword.arg]
                    if slot_index < len(positional_args):
                        raise ValueError(
                            f"Argument '{keyword.arg}' was provided both "
                            "positionally and by keyword"
                        )
                elif keyword.arg in kwonly_types:
                    expected_type = kwonly_types[keyword.arg]
                else:
                    raise NotImplementedError(
                        f"Keyword argument '{keyword.arg}' is not supported"
                    )
                if keyword_values is not None and keyword.arg in keyword_values:
                    value = keyword_values[keyword.arg]
                else:
                    value = self.require_value(keyword.value, self.visit(keyword.value))
                value = self.coerce_value_to_type(
                    value, expected_type, self._loc(keyword.value)
                )
                kwname_values.append(
                    py_ops.StrConstantOp(
                        self.get_py_type("!py.str"),
                        ir.StringAttr.get(keyword.arg, self.ctx),
                    ).result
                )
                kwvalue_values.append(self.ensure_object(value, loc=loc))

        required_positional_count = len(positional_param_types) - defaults_count
        for index, name in enumerate(positional_param_names):
            if index < len(positional_args):
                continue
            if name in seen_keywords:
                continue
            if index < required_positional_count:
                raise ValueError(f"Missing required argument '{name}'")

        for name in kwonly_names:
            if name in seen_keywords or name in kwdefault_name_set:
                continue
            raise ValueError(f"Missing required keyword-only argument '{name}'")

        return (
            posargs,
            self.build_tuple(kwname_values, loc=loc),
            self.build_tuple(kwvalue_values, loc=loc),
        )

    def _attach_returned_callable_metadata(
        self,
        op: ir.Operation,
        func_info: FunctionInfo,
    ) -> None:
        self._attach_returned_callable_info(op, func_info.returned_function_info)

    def _attach_returned_callable_info(
        self,
        op: ir.Operation,
        returned: FunctionInfo | None,
    ) -> None:
        if returned is None:
            return
        op.attributes["lython.returned_callable_symbol"] = ir.FlatSymbolRefAttr.get(
            returned.symbol, self.ctx
        )
        op.attributes["lython.returned_callable_defaults_count"] = ir.IntegerAttr.get(
            ir.IntegerType.get_signless(64, context=self.ctx),
            returned.defaults_count,
        )
        if returned.kwdefault_names:
            op.attributes["lython.returned_callable_kwdefault_names"] = ir.ArrayAttr.get(
                [ir.StringAttr.get(name, self.ctx) for name in returned.kwdefault_names],
                context=self.ctx,
            )

    def _materialize_known_callable_result(
        self,
        value: ir.Value,
        info: FunctionInfo | None,
        loc: ir.Location,
    ) -> ir.Value:
        if info is None or not str(value.type).startswith("!py.func<"):
            return value
        return self._build_python_callable(
            symbol=info.symbol,
            func_type=value.type,
            defaults=info.defaults,
            kwdefaults=info.kwdefaults,
            closure=info.closure,
            loc=loc,
            force_make_function=bool(
                info.defaults is not None
                or info.kwdefaults is not None
                or info.closure is not None
            ),
        )

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
    ) -> FunctionInfo | None:
        if returned_function_info is not None:
            return self._materialize_returned_callable_info_from_call(
                returned_function_info,
                positional_param_names=positional_param_names,
                kwonly_names=kwonly_names,
                positional_arg_values=positional_arg_values,
                keyword_arg_values=keyword_arg_values,
                loc=loc,
            )
        if returned_callable_arg_index is None:
            return None
        keyword_arg_nodes = keyword_arg_nodes or {}
        keyword_arg_values = keyword_arg_values or {}
        positional_param_names = list(positional_param_names)
        kwonly_names = list(kwonly_names)
        positional_default_callable_infos = list(positional_default_callable_infos)
        kwonly_default_callable_infos = list(kwonly_default_callable_infos)
        if returned_callable_arg_index < len(positional_arg_values):
            value = positional_arg_values[returned_callable_arg_index]
            node = (
                positional_arg_nodes[returned_callable_arg_index]
                if returned_callable_arg_index < len(positional_arg_nodes)
                else None
            )
            if node is None:
                return self.resolve_function_info_from_value(value)
            return self.resolve_function_info_from_expression(node, value)
        if returned_callable_arg_index < len(positional_param_names):
            param_name = positional_param_names[returned_callable_arg_index]
            if param_name in keyword_arg_values:
                value = keyword_arg_values[param_name]
                node = keyword_arg_nodes.get(param_name)
                if node is None:
                    return self.resolve_function_info_from_value(value)
                return self.resolve_function_info_from_expression(node, value)
            default_start = len(positional_param_names) - defaults_count
            if (
                defaults_count > 0
                and returned_callable_arg_index >= default_start
                and default_start >= 0
            ):
                default_index = returned_callable_arg_index - default_start
                if 0 <= default_index < len(positional_default_callable_infos):
                    return positional_default_callable_infos[default_index]
            return None
        else:
            kw_index = returned_callable_arg_index - len(positional_param_names)
            if kw_index < 0 or kw_index >= len(kwonly_names):
                return None
            param_name = kwonly_names[kw_index]
            if param_name in keyword_arg_values:
                value = keyword_arg_values[param_name]
                node = keyword_arg_nodes.get(param_name)
                if node is None:
                    return self.resolve_function_info_from_value(value)
                return self.resolve_function_info_from_expression(node, value)
            if 0 <= kw_index < len(kwonly_default_callable_infos):
                return kwonly_default_callable_infos[kw_index]
            return None

    def _resolve_argument_value_for_summary_index(
        self,
        *,
        parameter_index: int,
        positional_param_names: tuple[str, ...] | list[str],
        kwonly_names: tuple[str, ...] | list[str],
        positional_arg_values: list[ir.Value],
        keyword_arg_values: dict[str, ir.Value] | None = None,
    ) -> ir.Value | None:
        keyword_arg_values = keyword_arg_values or {}
        positional_param_names = list(positional_param_names)
        kwonly_names = list(kwonly_names)

        if parameter_index < len(positional_arg_values):
            return positional_arg_values[parameter_index]
        if parameter_index < len(positional_param_names):
            return keyword_arg_values.get(positional_param_names[parameter_index])
        kw_index = parameter_index - len(positional_param_names)
        if kw_index < 0 or kw_index >= len(kwonly_names):
            return None
        return keyword_arg_values.get(kwonly_names[kw_index])

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
    ) -> ir.Value | None:
        if cache is None:
            cache = {}
        cache_key = id(value)
        if cache_key in cache:
            return cache[cache_key]

        parameter_index = self.resolve_current_function_parameter_index_from_value(
            value
        )
        if parameter_index is not None:
            mapped = self._resolve_argument_value_for_summary_index(
                parameter_index=parameter_index,
                positional_param_names=positional_param_names,
                kwonly_names=kwonly_names,
                positional_arg_values=positional_arg_values,
                keyword_arg_values=keyword_arg_values,
            )
            if mapped is None and summary_info is not None:
                closure_base = len(summary_info.arg_types) + len(
                    summary_info.kwonly_arg_types
                )
                closure_slot = parameter_index - closure_base
                if 0 <= closure_slot < len(summary_info.closure_capture_arg_indices):
                    captured_param_index = summary_info.closure_capture_arg_indices[
                        closure_slot
                    ]
                    if captured_param_index is not None:
                        mapped = self._resolve_argument_value_for_summary_index(
                            parameter_index=captured_param_index,
                            positional_param_names=positional_param_names,
                            kwonly_names=kwonly_names,
                            positional_arg_values=positional_arg_values,
                            keyword_arg_values=keyword_arg_values,
                        )
            cache[cache_key] = mapped
            return mapped

        op = self._value_operation(value)
        if op is None:
            cache[cache_key] = value
            return value

        op_name = str(getattr(op, "name", ""))
        operands = list(getattr(op, "operands", ()))
        attributes = getattr(op, "attributes", {})

        if op_name in {"py.cast.identity", "py.publish"}:
            if not operands:
                cache[cache_key] = None
                return None
            remapped = self._remap_summary_value_to_callsite(
                operands[0],
                summary_info=summary_info,
                positional_param_names=positional_param_names,
                kwonly_names=kwonly_names,
                positional_arg_values=positional_arg_values,
                keyword_arg_values=keyword_arg_values,
                loc=loc,
                cache=cache,
            )
            cache[cache_key] = remapped
            return remapped

        with loc, self.insertion_point():
            result: ir.Value | None
            if op_name == "arith.constant":
                result = arith_ops.ConstantOp(value.type, attributes["value"]).result
            elif op_name == "py.str.constant":
                result = py_ops.StrConstantOp(value.type, attributes["value"]).result
            elif op_name == "py.int.constant":
                result = py_ops.IntConstantOp(value.type, attributes["value"]).result
            elif op_name == "py.float.constant":
                result = py_ops.FloatConstantOp(value.type, attributes["value"]).result
            elif op_name == "py.none":
                result = py_ops.NoneOp(value.type).result
            elif op_name == "py.tuple.empty":
                result = py_ops.TupleEmptyOp(value.type).result
            elif op_name == "py.tuple.create":
                remapped_elements = [
                    self._remap_summary_value_to_callsite(
                        operand,
                        summary_info=summary_info,
                        positional_param_names=positional_param_names,
                        kwonly_names=kwonly_names,
                        positional_arg_values=positional_arg_values,
                        keyword_arg_values=keyword_arg_values,
                        loc=loc,
                        cache=cache,
                    )
                    for operand in operands
                ]
                if any(element is None for element in remapped_elements):
                    result = None
                else:
                    result = py_ops.TupleCreateOp(value.type, remapped_elements).result
            elif op_name == "py.dict.empty":
                result = py_ops.DictEmptyOp(value.type).result
            elif op_name == "py.dict.insert":
                remapped_operands = [
                    self._remap_summary_value_to_callsite(
                        operand,
                        summary_info=summary_info,
                        positional_param_names=positional_param_names,
                        kwonly_names=kwonly_names,
                        positional_arg_values=positional_arg_values,
                        keyword_arg_values=keyword_arg_values,
                        loc=loc,
                        cache=cache,
                    )
                    for operand in operands
                ]
                if any(operand is None for operand in remapped_operands):
                    result = None
                else:
                    result = py_ops.DictInsertOp(
                        value.type,
                        remapped_operands[0],
                        remapped_operands[1],
                        remapped_operands[2],
                    ).result
            elif op_name == "py.upcast":
                if not operands:
                    result = None
                else:
                    remapped = self._remap_summary_value_to_callsite(
                        operands[0],
                        summary_info=summary_info,
                        positional_param_names=positional_param_names,
                        kwonly_names=kwonly_names,
                        positional_arg_values=positional_arg_values,
                        keyword_arg_values=keyword_arg_values,
                        loc=loc,
                        cache=cache,
                    )
                    result = (
                        py_ops.UpcastOp(value.type, remapped).result
                        if remapped is not None
                        else None
                    )
            elif op_name == "py.attr.get":
                if not operands:
                    result = None
                else:
                    remapped_object = self._remap_summary_value_to_callsite(
                        operands[0],
                        summary_info=summary_info,
                        positional_param_names=positional_param_names,
                        kwonly_names=kwonly_names,
                        positional_arg_values=positional_arg_values,
                        keyword_arg_values=keyword_arg_values,
                        loc=loc,
                        cache=cache,
                    )
                    result = (
                        py_ops.AttrGetOp(value.type, remapped_object, attributes["name"]).result
                        if remapped_object is not None
                        else None
                    )
            elif op_name == "py.list.get":
                remapped_list = self._remap_summary_value_to_callsite(
                    operands[0],
                    summary_info=summary_info,
                    positional_param_names=positional_param_names,
                    kwonly_names=kwonly_names,
                    positional_arg_values=positional_arg_values,
                    keyword_arg_values=keyword_arg_values,
                    loc=loc,
                    cache=cache,
                )
                remapped_index = self._remap_summary_value_to_callsite(
                    operands[1],
                    summary_info=summary_info,
                    positional_param_names=positional_param_names,
                    kwonly_names=kwonly_names,
                    positional_arg_values=positional_arg_values,
                    keyword_arg_values=keyword_arg_values,
                    loc=loc,
                    cache=cache,
                )
                if remapped_list is None or remapped_index is None:
                    result = None
                else:
                    result = py_ops.ListGetOp(value.type, remapped_list, remapped_index).result
            elif op_name == "py.func.object":
                result = py_ops.FuncObjectOp(value.type, attributes["target"]).result
            elif op_name == "py.make_function":
                (
                    defaults_operand,
                    kwdefaults_operand,
                    closure_operand,
                    annotations_operand,
                    module_operand,
                ) = self._extract_make_function_operands(op)

                def remap_optional(operand: ir.Value | None) -> ir.Value | None:
                    if operand is None:
                        return None
                    return self._remap_summary_value_to_callsite(
                        operand,
                        summary_info=summary_info,
                        positional_param_names=positional_param_names,
                        kwonly_names=kwonly_names,
                        positional_arg_values=positional_arg_values,
                        keyword_arg_values=keyword_arg_values,
                        loc=loc,
                        cache=cache,
                    )

                result = py_ops.MakeFunctionOp(
                    value.type,
                    attributes["target"],
                    defaults=remap_optional(defaults_operand),
                    kwdefaults=remap_optional(kwdefaults_operand),
                    closure=remap_optional(closure_operand),
                    annotations=remap_optional(annotations_operand),
                    module=remap_optional(module_operand),
                ).result
            elif op_name == "py.cast.to_prim":
                if not operands:
                    result = None
                else:
                    remapped_input = self._remap_summary_value_to_callsite(
                        operands[0],
                        summary_info=summary_info,
                        positional_param_names=positional_param_names,
                        kwonly_names=kwonly_names,
                        positional_arg_values=positional_arg_values,
                        keyword_arg_values=keyword_arg_values,
                        loc=loc,
                        cache=cache,
                    )
                    result = (
                        py_ops.CastToPrimOp(
                            value.type, remapped_input, attributes["mode"]
                        ).result
                        if remapped_input is not None
                        else None
                    )
            elif op_name == "py.cast.from_prim":
                if not operands:
                    result = None
                else:
                    remapped_input = self._remap_summary_value_to_callsite(
                        operands[0],
                        summary_info=summary_info,
                        positional_param_names=positional_param_names,
                        kwonly_names=kwonly_names,
                        positional_arg_values=positional_arg_values,
                        keyword_arg_values=keyword_arg_values,
                        loc=loc,
                        cache=cache,
                    )
                    result = (
                        py_ops.CastFromPrimOp(value.type, remapped_input).result
                        if remapped_input is not None
                        else None
                    )
            elif op_name == "py.num.add":
                remapped_lhs = self._remap_summary_value_to_callsite(
                    operands[0],
                    summary_info=summary_info,
                    positional_param_names=positional_param_names,
                    kwonly_names=kwonly_names,
                    positional_arg_values=positional_arg_values,
                    keyword_arg_values=keyword_arg_values,
                    loc=loc,
                    cache=cache,
                )
                remapped_rhs = self._remap_summary_value_to_callsite(
                    operands[1],
                    summary_info=summary_info,
                    positional_param_names=positional_param_names,
                    kwonly_names=kwonly_names,
                    positional_arg_values=positional_arg_values,
                    keyword_arg_values=keyword_arg_values,
                    loc=loc,
                    cache=cache,
                )
                if remapped_lhs is None or remapped_rhs is None:
                    result = None
                else:
                    result = py_ops.NumAddOp(remapped_lhs, remapped_rhs).result
            elif op_name == "py.num.sub":
                remapped_lhs = self._remap_summary_value_to_callsite(
                    operands[0],
                    summary_info=summary_info,
                    positional_param_names=positional_param_names,
                    kwonly_names=kwonly_names,
                    positional_arg_values=positional_arg_values,
                    keyword_arg_values=keyword_arg_values,
                    loc=loc,
                    cache=cache,
                )
                remapped_rhs = self._remap_summary_value_to_callsite(
                    operands[1],
                    summary_info=summary_info,
                    positional_param_names=positional_param_names,
                    kwonly_names=kwonly_names,
                    positional_arg_values=positional_arg_values,
                    keyword_arg_values=keyword_arg_values,
                    loc=loc,
                    cache=cache,
                )
                if remapped_lhs is None or remapped_rhs is None:
                    result = None
                else:
                    result = py_ops.NumSubOp(remapped_lhs, remapped_rhs).result
            elif op_name in {
                "py.num.eq",
                "py.num.ne",
                "py.num.lt",
                "py.num.le",
                "py.num.gt",
                "py.num.ge",
            }:
                remapped_lhs = self._remap_summary_value_to_callsite(
                    operands[0],
                    summary_info=summary_info,
                    positional_param_names=positional_param_names,
                    kwonly_names=kwonly_names,
                    positional_arg_values=positional_arg_values,
                    keyword_arg_values=keyword_arg_values,
                    loc=loc,
                    cache=cache,
                )
                remapped_rhs = self._remap_summary_value_to_callsite(
                    operands[1],
                    summary_info=summary_info,
                    positional_param_names=positional_param_names,
                    kwonly_names=kwonly_names,
                    positional_arg_values=positional_arg_values,
                    keyword_arg_values=keyword_arg_values,
                    loc=loc,
                    cache=cache,
                )
                if remapped_lhs is None or remapped_rhs is None:
                    result = None
                elif op_name == "py.num.eq":
                    result = py_ops.NumEqOp(value.type, remapped_lhs, remapped_rhs).result
                elif op_name == "py.num.ne":
                    result = py_ops.NumNeOp(value.type, remapped_lhs, remapped_rhs).result
                elif op_name == "py.num.lt":
                    result = py_ops.NumLtOp(value.type, remapped_lhs, remapped_rhs).result
                elif op_name == "py.num.le":
                    result = py_ops.NumLeOp(value.type, remapped_lhs, remapped_rhs).result
                elif op_name == "py.num.gt":
                    result = py_ops.NumGtOp(value.type, remapped_lhs, remapped_rhs).result
                else:
                    result = py_ops.NumGeOp(value.type, remapped_lhs, remapped_rhs).result
            elif op_name == "arith.cmpi":
                remapped_lhs = self._remap_summary_value_to_callsite(
                    operands[0],
                    summary_info=summary_info,
                    positional_param_names=positional_param_names,
                    kwonly_names=kwonly_names,
                    positional_arg_values=positional_arg_values,
                    keyword_arg_values=keyword_arg_values,
                    loc=loc,
                    cache=cache,
                )
                remapped_rhs = self._remap_summary_value_to_callsite(
                    operands[1],
                    summary_info=summary_info,
                    positional_param_names=positional_param_names,
                    kwonly_names=kwonly_names,
                    positional_arg_values=positional_arg_values,
                    keyword_arg_values=keyword_arg_values,
                    loc=loc,
                    cache=cache,
                )
                if remapped_lhs is None or remapped_rhs is None:
                    result = None
                else:
                    result = arith_ops.CmpIOp(
                        attributes["predicate"], remapped_lhs, remapped_rhs
                    ).result
            elif op_name == "arith.cmpf":
                remapped_lhs = self._remap_summary_value_to_callsite(
                    operands[0],
                    summary_info=summary_info,
                    positional_param_names=positional_param_names,
                    kwonly_names=kwonly_names,
                    positional_arg_values=positional_arg_values,
                    keyword_arg_values=keyword_arg_values,
                    loc=loc,
                    cache=cache,
                )
                remapped_rhs = self._remap_summary_value_to_callsite(
                    operands[1],
                    summary_info=summary_info,
                    positional_param_names=positional_param_names,
                    kwonly_names=kwonly_names,
                    positional_arg_values=positional_arg_values,
                    keyword_arg_values=keyword_arg_values,
                    loc=loc,
                    cache=cache,
                )
                if remapped_lhs is None or remapped_rhs is None:
                    result = None
                else:
                    result = arith_ops.CmpFOp(
                        attributes["predicate"], remapped_lhs, remapped_rhs
                    ).result
            elif op_name in {
                "arith.addi",
                "arith.subi",
                "arith.muli",
                "arith.divsi",
                "arith.remsi",
                "arith.addf",
                "arith.subf",
                "arith.mulf",
                "arith.divf",
                "arith.remf",
            }:
                remapped_lhs = self._remap_summary_value_to_callsite(
                    operands[0],
                    summary_info=summary_info,
                    positional_param_names=positional_param_names,
                    kwonly_names=kwonly_names,
                    positional_arg_values=positional_arg_values,
                    keyword_arg_values=keyword_arg_values,
                    loc=loc,
                    cache=cache,
                )
                remapped_rhs = self._remap_summary_value_to_callsite(
                    operands[1],
                    summary_info=summary_info,
                    positional_param_names=positional_param_names,
                    kwonly_names=kwonly_names,
                    positional_arg_values=positional_arg_values,
                    keyword_arg_values=keyword_arg_values,
                    loc=loc,
                    cache=cache,
                )
                if remapped_lhs is None or remapped_rhs is None:
                    result = None
                elif op_name == "arith.addi":
                    result = arith_ops.AddIOp(remapped_lhs, remapped_rhs).result
                elif op_name == "arith.subi":
                    result = arith_ops.SubIOp(remapped_lhs, remapped_rhs).result
                elif op_name == "arith.muli":
                    result = arith_ops.MulIOp(remapped_lhs, remapped_rhs).result
                elif op_name == "arith.divsi":
                    result = arith_ops.DivSIOp(remapped_lhs, remapped_rhs).result
                elif op_name == "arith.remsi":
                    result = arith_ops.RemSIOp(remapped_lhs, remapped_rhs).result
                elif op_name == "arith.addf":
                    result = arith_ops.AddFOp(remapped_lhs, remapped_rhs).result
                elif op_name == "arith.subf":
                    result = arith_ops.SubFOp(remapped_lhs, remapped_rhs).result
                elif op_name == "arith.mulf":
                    result = arith_ops.MulFOp(remapped_lhs, remapped_rhs).result
                elif op_name == "arith.divf":
                    result = arith_ops.DivFOp(remapped_lhs, remapped_rhs).result
                else:
                    result = arith_ops.RemFOp(remapped_lhs, remapped_rhs).result
            else:
                result = None

        cache[cache_key] = result
        return result

    def _materialize_returned_callable_info_from_call(
        self,
        info: FunctionInfo,
        *,
        positional_param_names: tuple[str, ...] | list[str],
        kwonly_names: tuple[str, ...] | list[str],
        positional_arg_values: list[ir.Value],
        keyword_arg_values: dict[str, ir.Value] | None = None,
        loc: ir.Location,
    ) -> FunctionInfo:
        keyword_arg_values = keyword_arg_values or {}
        attempted_defaults = info.defaults is not None
        attempted_kwdefaults = info.kwdefaults is not None
        attempted_closure = info.closure is not None
        remapped_defaults = (
            self._remap_summary_value_to_callsite(
                info.defaults,
                summary_info=info,
                positional_param_names=positional_param_names,
                kwonly_names=kwonly_names,
                positional_arg_values=positional_arg_values,
                keyword_arg_values=keyword_arg_values,
                loc=loc,
            )
            if info.defaults is not None
            else None
        )
        remapped_kwdefaults = (
            self._remap_summary_value_to_callsite(
                info.kwdefaults,
                summary_info=info,
                positional_param_names=positional_param_names,
                kwonly_names=kwonly_names,
                positional_arg_values=positional_arg_values,
                keyword_arg_values=keyword_arg_values,
                loc=loc,
            )
            if info.kwdefaults is not None
            else None
        )
        remapped_closure = (
            self._remap_summary_value_to_callsite(
                info.closure,
                summary_info=info,
                positional_param_names=positional_param_names,
                kwonly_names=kwonly_names,
                positional_arg_values=positional_arg_values,
                keyword_arg_values=keyword_arg_values,
                loc=loc,
            )
            if info.closure is not None
            else None
        )
        remapped_or_rebuilt_closure = remapped_closure

        if remapped_or_rebuilt_closure is None and info.closure_capture_arg_indices:
            closure_values: list[ir.Value] = []
            for parameter_index in info.closure_capture_arg_indices:
                if parameter_index is None:
                    closure_values = []
                    break
                value = self._resolve_argument_value_for_summary_index(
                    parameter_index=parameter_index,
                    positional_param_names=positional_param_names,
                    kwonly_names=kwonly_names,
                    positional_arg_values=positional_arg_values,
                    keyword_arg_values=keyword_arg_values,
                )
                if value is None:
                    closure_values = []
                    break
                closure_values.append(value)
            if closure_values:
                remapped_or_rebuilt_closure = self.build_tuple(closure_values, loc=loc)

        if (
            remapped_defaults is not None
            or remapped_kwdefaults is not None
            or remapped_or_rebuilt_closure is not None
        ):
            return info._replace(
                defaults=remapped_defaults if attempted_defaults else None,
                kwdefaults=remapped_kwdefaults if attempted_kwdefaults else None,
                closure=(
                    remapped_or_rebuilt_closure if attempted_closure else None
                ),
            )
        return info

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
    ) -> ir.Value:
        with loc, self.insertion_point():
            if (
                not force_make_function
                and defaults is None
                and kwdefaults is None
                and closure is None
            ):
                return py_ops.FuncObjectOp(func_type, symbol).result
            return py_ops.MakeFunctionOp(
                func_type,
                ir.FlatSymbolRefAttr.get(symbol, self.ctx),
                defaults=(
                    self._clone_bound_callable_metadata(defaults, loc)
                    if defaults is not None
                    else None
                ),
                kwdefaults=(
                    self._clone_bound_callable_metadata(kwdefaults, loc)
                    if kwdefaults is not None
                    else None
                ),
                closure=(
                    self._clone_bound_callable_metadata(closure, loc)
                    if closure is not None
                    else None
                ),
            ).result

    def _build_method_callable(
        self,
        *,
        symbol: str,
        func_type: ir.Type,
        defaults: ir.Value | None,
        kwdefaults: ir.Value | None,
        loc: ir.Location,
        force_make_function: bool = False,
    ) -> ir.Value:
        return self._build_python_callable(
            symbol=symbol,
            func_type=func_type,
            defaults=defaults,
            kwdefaults=kwdefaults,
            closure=None,
            loc=loc,
            force_make_function=force_make_function,
        )

    def _needs_keyword_callable_materialization(self, value: ir.Value) -> bool:
        current = value
        while True:
            op = self._value_operation(current)
            if op is None:
                return False
            op_name = str(getattr(op, "name", ""))
            if op_name == "py.cast.identity":
                operands = list(getattr(op, "operands", ()))
                if not operands:
                    return False
                current = operands[0]
                continue
            return op_name == "py.func.object"

    def _clone_bound_callable_metadata(
        self, value: ir.Value, loc: ir.Location
    ) -> ir.Value:
        current_parent = (
            getattr(self.current_block, "owner", None)
            if self.current_block is not None
            else None
        )
        owner = getattr(value, "owner", None)
        if current_parent is not None and owner is not None:
            owner_owner = getattr(owner, "owner", None)
            if owner_owner is not None and owner_owner == current_parent:
                return value
            owner_parent = getattr(owner, "parent", None)
            if owner_parent is not None and owner_parent == current_parent:
                return value
        opview = getattr(owner, "opview", None)
        if opview is None and owner is not None and not hasattr(owner, "operands"):
            return value
        op = getattr(opview, "operation", owner)
        if op is None:
            return value
        op_name = str(getattr(op, "name", ""))
        operands = list(getattr(op, "operands", []))
        attributes = getattr(op, "attributes", {})

        with loc, self.insertion_point():
            if op_name in {"py.cast.identity", "py.publish"}:
                info = self.resolve_function_info_from_value(value)
                if info is not None:
                    return self._build_python_callable(
                        symbol=info.symbol,
                        func_type=value.type,
                        defaults=info.defaults,
                        kwdefaults=info.kwdefaults,
                        closure=info.closure,
                        loc=loc,
                        force_make_function=bool(
                            info.defaults is not None
                            or info.kwdefaults is not None
                            or info.closure is not None
                        ),
                    )
                if operands:
                    return self._clone_bound_callable_metadata(operands[0], loc)
            if op_name == "py.str.constant":
                return py_ops.StrConstantOp(value.type, attributes["value"]).result
            if op_name == "py.int.constant":
                return py_ops.IntConstantOp(value.type, attributes["value"]).result
            if op_name == "py.float.constant":
                return py_ops.FloatConstantOp(value.type, attributes["value"]).result
            if op_name == "arith.constant":
                return arith_ops.ConstantOp(value.type, attributes["value"]).result
            if op_name == "py.none":
                return py_ops.NoneOp(value.type).result
            if op_name == "py.tuple.empty":
                return py_ops.TupleEmptyOp(value.type).result
            if op_name == "py.tuple.create":
                elements = [
                    self._clone_bound_callable_metadata(element, loc)
                    for element in operands
                ]
                return py_ops.TupleCreateOp(value.type, elements).result
            if op_name == "py.dict.empty":
                return py_ops.DictEmptyOp(value.type).result
            if op_name == "py.dict.insert":
                base = self._clone_bound_callable_metadata(operands[0], loc)
                key = self._clone_bound_callable_metadata(operands[1], loc)
                item = self._clone_bound_callable_metadata(operands[2], loc)
                return py_ops.DictInsertOp(value.type, base, key, item).result
            if op_name == "py.upcast":
                cloned = self._clone_bound_callable_metadata(operands[0], loc)
                return py_ops.UpcastOp(value.type, cloned).result
            if op_name == "py.attr.get":
                cloned_object = self._clone_bound_callable_metadata(operands[0], loc)
                return py_ops.AttrGetOp(
                    value.type,
                    cloned_object,
                    attributes["name"],
                ).result
            if op_name == "py.list.get":
                cloned_list = self._clone_bound_callable_metadata(operands[0], loc)
                cloned_index = self._clone_bound_callable_metadata(operands[1], loc)
                return py_ops.ListGetOp(value.type, cloned_list, cloned_index).result
            if op_name == "py.func.object":
                return py_ops.FuncObjectOp(value.type, attributes["target"]).result
            if op_name == "py.make_function":
                segment_sizes = [int(size) for size in attributes["operandSegmentSizes"]]
                segment_index = 0
                operand_index = 0

                def take_optional() -> ir.Value | None:
                    nonlocal segment_index, operand_index
                    size = segment_sizes[segment_index]
                    result = operands[operand_index] if size else None
                    segment_index += 1
                    operand_index += size
                    return result

                defaults_operand = take_optional()
                kwdefaults_operand = take_optional()
                closure_operand = take_optional()
                annotations_operand = take_optional()
                module_operand = take_optional()
                defaults = (
                    self._clone_bound_callable_metadata(defaults_operand, loc)
                    if defaults_operand is not None
                    else None
                )
                kwdefaults = (
                    self._clone_bound_callable_metadata(kwdefaults_operand, loc)
                    if kwdefaults_operand is not None
                    else None
                )
                closure = (
                    self._clone_bound_callable_metadata(closure_operand, loc)
                    if closure_operand is not None
                    else None
                )
                annotations = (
                    self._clone_bound_callable_metadata(annotations_operand, loc)
                    if annotations_operand is not None
                    else None
                )
                module = (
                    self._clone_bound_callable_metadata(module_operand, loc)
                    if module_operand is not None
                    else None
                )
                return py_ops.MakeFunctionOp(
                    value.type,
                    attributes["target"],
                    defaults=defaults,
                    kwdefaults=kwdefaults,
                    closure=closure,
                    annotations=annotations,
                    module=module,
                ).result
            if op_name == "py.cast.to_prim":
                cloned_input = self._clone_bound_callable_metadata(operands[0], loc)
                return py_ops.CastToPrimOp(
                    value.type,
                    cloned_input,
                    attributes["mode"],
                ).result
            if op_name == "py.cast.from_prim":
                cloned_input = self._clone_bound_callable_metadata(operands[0], loc)
                return py_ops.CastFromPrimOp(value.type, cloned_input).result
            if op_name == "py.num.add":
                lhs = self._clone_bound_callable_metadata(operands[0], loc)
                rhs = self._clone_bound_callable_metadata(operands[1], loc)
                return py_ops.NumAddOp(lhs, rhs).result
            if op_name == "py.num.sub":
                lhs = self._clone_bound_callable_metadata(operands[0], loc)
                rhs = self._clone_bound_callable_metadata(operands[1], loc)
                return py_ops.NumSubOp(lhs, rhs).result
            if op_name == "py.num.eq":
                lhs = self._clone_bound_callable_metadata(operands[0], loc)
                rhs = self._clone_bound_callable_metadata(operands[1], loc)
                return py_ops.NumEqOp(value.type, lhs, rhs).result
            if op_name == "py.num.ne":
                lhs = self._clone_bound_callable_metadata(operands[0], loc)
                rhs = self._clone_bound_callable_metadata(operands[1], loc)
                return py_ops.NumNeOp(value.type, lhs, rhs).result
            if op_name == "py.num.lt":
                lhs = self._clone_bound_callable_metadata(operands[0], loc)
                rhs = self._clone_bound_callable_metadata(operands[1], loc)
                return py_ops.NumLtOp(value.type, lhs, rhs).result
            if op_name == "py.num.le":
                lhs = self._clone_bound_callable_metadata(operands[0], loc)
                rhs = self._clone_bound_callable_metadata(operands[1], loc)
                return py_ops.NumLeOp(value.type, lhs, rhs).result
            if op_name == "py.num.gt":
                lhs = self._clone_bound_callable_metadata(operands[0], loc)
                rhs = self._clone_bound_callable_metadata(operands[1], loc)
                return py_ops.NumGtOp(value.type, lhs, rhs).result
            if op_name == "py.num.ge":
                lhs = self._clone_bound_callable_metadata(operands[0], loc)
                rhs = self._clone_bound_callable_metadata(operands[1], loc)
                return py_ops.NumGeOp(value.type, lhs, rhs).result
            if op_name == "arith.cmpi":
                lhs = self._clone_bound_callable_metadata(operands[0], loc)
                rhs = self._clone_bound_callable_metadata(operands[1], loc)
                return arith_ops.CmpIOp(attributes["predicate"], lhs, rhs).result
            if op_name == "arith.cmpf":
                lhs = self._clone_bound_callable_metadata(operands[0], loc)
                rhs = self._clone_bound_callable_metadata(operands[1], loc)
                return arith_ops.CmpFOp(attributes["predicate"], lhs, rhs).result
            if op_name == "arith.addi":
                lhs = self._clone_bound_callable_metadata(operands[0], loc)
                rhs = self._clone_bound_callable_metadata(operands[1], loc)
                return arith_ops.AddIOp(lhs, rhs).result
            if op_name == "arith.subi":
                lhs = self._clone_bound_callable_metadata(operands[0], loc)
                rhs = self._clone_bound_callable_metadata(operands[1], loc)
                return arith_ops.SubIOp(lhs, rhs).result
            if op_name == "arith.muli":
                lhs = self._clone_bound_callable_metadata(operands[0], loc)
                rhs = self._clone_bound_callable_metadata(operands[1], loc)
                return arith_ops.MulIOp(lhs, rhs).result
            if op_name == "arith.divsi":
                lhs = self._clone_bound_callable_metadata(operands[0], loc)
                rhs = self._clone_bound_callable_metadata(operands[1], loc)
                return arith_ops.DivSIOp(lhs, rhs).result
            if op_name == "arith.remsi":
                lhs = self._clone_bound_callable_metadata(operands[0], loc)
                rhs = self._clone_bound_callable_metadata(operands[1], loc)
                return arith_ops.RemSIOp(lhs, rhs).result
            if op_name == "arith.addf":
                lhs = self._clone_bound_callable_metadata(operands[0], loc)
                rhs = self._clone_bound_callable_metadata(operands[1], loc)
                return arith_ops.AddFOp(lhs, rhs).result
            if op_name == "arith.subf":
                lhs = self._clone_bound_callable_metadata(operands[0], loc)
                rhs = self._clone_bound_callable_metadata(operands[1], loc)
                return arith_ops.SubFOp(lhs, rhs).result
            if op_name == "arith.mulf":
                lhs = self._clone_bound_callable_metadata(operands[0], loc)
                rhs = self._clone_bound_callable_metadata(operands[1], loc)
                return arith_ops.MulFOp(lhs, rhs).result
            if op_name == "arith.divf":
                lhs = self._clone_bound_callable_metadata(operands[0], loc)
                rhs = self._clone_bound_callable_metadata(operands[1], loc)
                return arith_ops.DivFOp(lhs, rhs).result
            if op_name == "arith.remf":
                lhs = self._clone_bound_callable_metadata(operands[0], loc)
                rhs = self._clone_bound_callable_metadata(operands[1], loc)
                return arith_ops.RemFOp(lhs, rhs).result

        raise NotImplementedError(
            f"Cloning bound callable metadata from {op_name or type(opview).__name__} "
            "is not supported yet"
        )

    def visit_Name(self, node: ast.Name) -> ir.Value:
        if isinstance(node.ctx, ast.Store):
            raise NotImplementedError("Store context handled elsewhere")

        # In native mode, check if this is a primitive constant that needs
        # to be recreated locally (due to region isolation)
        if self._in_native_func:
            prim_const = self.get_prim_constant(node.id)
            if prim_const is not None:
                mlir_type, value = prim_const
                with self._loc(node), self.insertion_point():
                    # Recreate the constant locally in this function's region
                    if isinstance(value, int):
                        attr = ir.IntegerAttr.get(mlir_type, value)
                    else:
                        attr = ir.FloatAttr.get(mlir_type, value)
                    return arith_ops.ConstantOp(mlir_type, attr).result

        for depth, scope in enumerate(reversed(self._scope_stack)):
            if node.id not in scope:
                continue
            value = scope[node.id]
            if depth == 0 or not str(value.type).startswith("!py.func<"):
                return value
            try:
                return self._clone_bound_callable_metadata(value, self._loc(node))
            except NotImplementedError:
                pass
            try:
                info = self.lookup_function_binding(node.id)
            except NameError:
                info = None
            if info is not None:
                return self._build_python_callable(
                    symbol=info.symbol,
                    func_type=value.type,
                    defaults=info.defaults,
                    kwdefaults=info.kwdefaults,
                    closure=info.closure,
                    loc=self._loc(node),
                    force_make_function=bool(
                        info.defaults is not None
                        or info.kwdefaults is not None
                        or info.closure is not None
                    ),
                )
            return self._clone_bound_callable_metadata(value, self._loc(node))
        try:
            func_info = self.lookup_function(node.id)
        except NameError as exc:
            raise NameError(f"Variable reference '{node.id}' not implemented") from exc
        with self._loc(node), self.insertion_point():
            symbol = ir.FlatSymbolRefAttr.get(func_info.symbol, self.ctx)
            return py_ops.FuncObjectOp(func_info.func_type, symbol).result

    def visit_BinOp(self, node: ast.BinOp) -> ir.Value:
        """
        二項演算を処理する

        ```asdl
        BinOp(expr left, operator op, expr right)
        ```
        """
        lhs = self.require_value(node.left, self.visit(node.left))
        rhs = self.require_value(node.right, self.visit(node.right))
        lhs, rhs = self.coerce_operands_for_binary(lhs, rhs, self._loc(node))

        # In native mode or for primitive types, use arith.* operations
        if self._in_native_func or (
            not self.is_py_type(lhs.type) and not self.is_py_type(rhs.type)
        ):
            return self._handle_primitive_binop(node.op, lhs, rhs, self._loc(node))

        # In object mode, use py.num.* operations
        with self._loc(node), self.insertion_point():
            if isinstance(node.op, ast.Add):
                return py_ops.NumAddOp(lhs, rhs).result
            if isinstance(node.op, ast.Sub):
                return py_ops.NumSubOp(lhs, rhs).result
        raise NotImplementedError("Unsupported binary operation")

    def _handle_primitive_binop(
        self, op: ast.operator, lhs: ir.Value, rhs: ir.Value, loc: ir.Location
    ) -> ir.Value:
        """Handle binary operation on primitive types using arith dialect."""
        lhs_type = lhs.type
        is_float = self.is_primitive_float_type(lhs_type)
        is_int = self.is_primitive_int_type(lhs_type)

        with loc, self.insertion_point():
            if isinstance(op, ast.Add):
                if is_float:
                    return arith_ops.AddFOp(lhs, rhs).result
                if is_int:
                    return arith_ops.AddIOp(lhs, rhs).result
                raise NotImplementedError("Unsupported primitive add type")
            elif isinstance(op, ast.Sub):
                if is_float:
                    return arith_ops.SubFOp(lhs, rhs).result
                if is_int:
                    return arith_ops.SubIOp(lhs, rhs).result
                raise NotImplementedError("Unsupported primitive sub type")
            elif isinstance(op, ast.Mult):
                if is_float:
                    return arith_ops.MulFOp(lhs, rhs).result
                if is_int:
                    return arith_ops.MulIOp(lhs, rhs).result
                raise NotImplementedError("Unsupported primitive mul type")
            elif isinstance(op, ast.FloorDiv):
                if is_float:
                    raise NotImplementedError("Floor division on floats not supported")
                if is_int:
                    return arith_ops.DivSIOp(lhs, rhs).result
                raise NotImplementedError("Unsupported primitive div type")
            elif isinstance(op, ast.Div):
                if is_float:
                    if isinstance(lhs_type, ir.ShapedType):
                        raise NotImplementedError(
                            "Division is not supported for vector/matrix/tensor types"
                        )
                    return arith_ops.DivFOp(lhs, rhs).result
                if is_int:
                    if isinstance(lhs_type, ir.ShapedType):
                        raise NotImplementedError(
                            "Division is not supported for vector/matrix/tensor types"
                        )
                    return arith_ops.DivSIOp(lhs, rhs).result
                raise NotImplementedError("Unsupported primitive div type")
            elif isinstance(op, ast.MatMult):
                if not isinstance(lhs_type, ir.RankedTensorType) or not isinstance(
                    rhs.type, ir.RankedTensorType
                ):
                    raise NotImplementedError("Matrix multiplication requires tensors")
                if not self.is_primitive_float_type(lhs_type):
                    raise NotImplementedError(
                        "Matrix multiplication supports float tensors only"
                    )
                if lhs_type.rank != 2 or rhs.type.rank != 2:
                    raise NotImplementedError(
                        "Matrix multiplication supports rank-2 tensors"
                    )

                lhs_shape = list(lhs_type.shape)
                rhs_shape = list(rhs.type.shape)
                if (
                    ir.ShapedType.get_dynamic_size() in lhs_shape
                    or ir.ShapedType.get_dynamic_size() in rhs_shape
                ):
                    raise NotImplementedError("Dynamic shapes are not supported yet")
                if lhs_shape[1] != rhs_shape[0]:
                    raise ValueError("Matrix multiplication shape mismatch")

                elem_type = lhs_type.element_type
                zero = self._build_primitive_scalar(0.0, elem_type, loc)
                init = tensor_ops.EmptyOp(
                    [lhs_shape[0], rhs_shape[1]], elem_type
                ).result
                filled = self._build_linalg_fill(zero, init, elem_type, loc)
                return self._build_linalg_matmul(lhs, rhs, filled, elem_type, loc)
            elif isinstance(op, ast.Mod):
                if is_float:
                    return arith_ops.RemFOp(lhs, rhs).result
                if is_int:
                    return arith_ops.RemSIOp(lhs, rhs).result
                raise NotImplementedError("Unsupported primitive rem type")
            else:
                raise NotImplementedError(
                    f"Unsupported primitive binary operation: {type(op).__name__}"
                )

    def visit_UnaryOp(self, node: ast.UnaryOp) -> None:
        """
        単項演算を処理する

        ```asdl
        UnaryOp(unaryop op, expr operand)
        ```
        """
        operand = self.require_value(node.operand, self.visit(node.operand))
        operand_type = operand.type
        loc = self._loc(node)

        if isinstance(node.op, ast.UAdd):
            if self.is_py_type(operand_type):
                if operand_type in {
                    self.get_py_type("!py.int"),
                    self.get_py_type("!py.float"),
                }:
                    return operand
            elif not isinstance(operand_type, ir.ShapedType) and (
                self.is_primitive_int_type(operand_type)
                or self.is_primitive_float_type(operand_type)
            ):
                return operand
            raise NotImplementedError(
                f"Unary plus is not supported for {operand_type}"
            )

        if isinstance(node.op, ast.USub):
            with loc, self.insertion_point():
                if operand_type == self.get_py_type("!py.int"):
                    zero = py_ops.IntConstantOp(operand_type, "0").result
                    return py_ops.NumSubOp(zero, operand).result
                if operand_type == self.get_py_type("!py.float"):
                    zero = py_ops.FloatConstantOp(operand_type, 0.0).result
                    return py_ops.NumSubOp(zero, operand).result
                if isinstance(operand_type, ir.ShapedType):
                    raise NotImplementedError(
                        f"Unary minus is not supported for shaped primitive type {operand_type}"
                    )
                if self.is_primitive_float_type(operand_type):
                    zero = self._build_primitive_scalar(0.0, operand_type, loc)
                    return arith_ops.SubFOp(zero, operand).result
                if self.is_primitive_int_type(operand_type):
                    zero = self._build_primitive_scalar(0, operand_type, loc)
                    return arith_ops.SubIOp(zero, operand).result
            raise NotImplementedError(
                f"Unary minus is not supported for {operand_type}"
            )

        if isinstance(node.op, ast.Not):
            bool_type = self.get_py_type("!py.bool")
            i1_type = ir.IntegerType.get_signless(1, context=self.ctx)
            i64_type = ir.IntegerType.get_signless(64, context=self.ctx)
            f64_type = ir.F64Type.get(context=self.ctx)
            with loc, self.insertion_point():
                if operand_type == bool_type:
                    prim_value = py_ops.CastToPrimOp(
                        i1_type,
                        operand,
                        ir.StringAttr.get("exact", self.ctx),
                    ).result
                    zero = arith_ops.ConstantOp(i1_type, 0).result
                    prim_result = arith_ops.CmpIOp(
                        arith_ops.CmpIPredicate.eq, prim_value, zero
                    ).result
                    return py_ops.CastFromPrimOp(bool_type, prim_result).result
                if operand_type == self.get_py_type("!py.int"):
                    prim_value = py_ops.CastToPrimOp(
                        i64_type,
                        operand,
                        ir.StringAttr.get("exact", self.ctx),
                    ).result
                    zero = arith_ops.ConstantOp(i64_type, 0).result
                    prim_result = arith_ops.CmpIOp(
                        arith_ops.CmpIPredicate.eq, prim_value, zero
                    ).result
                    return py_ops.CastFromPrimOp(bool_type, prim_result).result
                if operand_type == self.get_py_type("!py.float"):
                    prim_value = py_ops.CastToPrimOp(
                        f64_type,
                        operand,
                        ir.StringAttr.get("exact", self.ctx),
                    ).result
                    zero = arith_ops.ConstantOp(f64_type, 0.0).result
                    prim_result = arith_ops.CmpFOp(
                        arith_ops.CmpFPredicate.OEQ, prim_value, zero
                    ).result
                    return py_ops.CastFromPrimOp(bool_type, prim_result).result
                if isinstance(operand_type, ir.ShapedType):
                    raise NotImplementedError(
                        f"Logical not is not supported for shaped primitive type {operand_type}"
                    )
                if self.is_primitive_float_type(operand_type):
                    zero = self._build_primitive_scalar(0.0, operand_type, loc)
                    return arith_ops.CmpFOp(
                        arith_ops.CmpFPredicate.OEQ, operand, zero
                    ).result
                if self.is_primitive_int_type(operand_type):
                    zero = self._build_primitive_scalar(0, operand_type, loc)
                    return arith_ops.CmpIOp(
                        arith_ops.CmpIPredicate.eq, operand, zero
                    ).result
            raise NotImplementedError(
                f"Logical not is not supported for {operand_type}"
            )

        raise NotImplementedError(
            f"Unsupported unary operation: {type(node.op).__name__}"
        )

    def visit_Lambda(self, node: ast.Lambda) -> None:
        """
        ラムダ式を処理する

        ```asdl
        Lambda(arguments args, expr body)
        ```
        """
        raise NotImplementedError("Lambda expression not implemented")

    def visit_IfExp(self, node: ast.IfExp) -> None:
        """
        三項演算子を処理する

        ```asdl
        IfExp(expr test, expr body, expr orelse)
        ```
        """
        raise NotImplementedError("If expression not implemented")

    def visit_Dict(self, node: ast.Dict) -> None:
        """
        辞書リテラルを処理する。
        PyDictObject*を生成し、キーと値のペアを追加する。

        ```asdl
        Dict(expr* keys, expr* values)
        ```
        """
        if not node.keys:
            raise NotImplementedError("Empty dict literal is not supported yet")

        entries = []
        for key_node, value_node in zip(node.keys, node.values):
            if key_node is None:
                raise NotImplementedError("Dict unpacking is not supported yet")
            key = self.require_value(key_node, self.visit(key_node))
            value = self.require_value(value_node, self.visit(value_node))
            entries.append((key, value))

        key_type = entries[0][0].type
        value_type = entries[0][1].type
        dict_type = self.get_py_type(f"!py.dict<{key_type}, {value_type}>")

        with self._loc(node), self.insertion_point():
            current = py_ops.DictEmptyOp(dict_type).result

        for key, value in entries:
            if key.type != key_type:
                key = self.coerce_value_to_type(key, key_type, self._loc(node))
            if value.type != value_type:
                value = self.coerce_value_to_type(value, value_type, self._loc(node))
            with self._loc(node), self.insertion_point():
                current = py_ops.DictInsertOp(dict_type, current, key, value).result
        return current

    def visit_Set(self, node: ast.Set) -> None:
        """
        集合リテラル

        ```asdl
        Set(expr* elts)
        ```
        """
        raise NotImplementedError("Set literal not implemented")

    def visit_ListComp(self, node: ast.ListComp) -> None:
        """
        リスト内包表記を処理する

        ```asdl
        ListComp(expr elt, comprehension* generators)
        ```
        """
        raise NotImplementedError("List comprehension not implemented")

    def visit_Constant(self, node: ast.Constant) -> ir.Value:
        with self._loc(node), self.insertion_point():
            if node.value is None:
                return py_ops.NoneOp(self.get_py_type("!py.none")).result
            if isinstance(node.value, bool):
                result_type = self.get_py_type("!py.bool")
                prim_type = ir.IntegerType.get_signless(1, context=self.ctx)
                prim_attr = ir.IntegerAttr.get(prim_type, int(node.value))
                prim_value = arith_ops.ConstantOp(prim_type, prim_attr).result
                return py_ops.CastFromPrimOp(result_type, prim_value).result
            if isinstance(node.value, int):
                result_type = self.get_py_type("!py.int")
                # Convert to string for arbitrary precision support
                return py_ops.IntConstantOp(result_type, str(node.value)).result
            if isinstance(node.value, float):
                result_type = self.get_py_type("!py.float")
                return py_ops.FloatConstantOp(result_type, node.value).result
            if isinstance(node.value, str):
                result_type = self.get_py_type("!py.str")
                attr = ir.StringAttr.get(node.value, self.ctx)
                return py_ops.StrConstantOp(result_type, attr).result
        raise NotImplementedError(f"Unsupported constant {node.value!r}")

    def visit_SetComp(self, node: ast.SetComp) -> None:
        """
        集合内包表記を処理する

        ```asdl
        SetComp(expr elt, comprehension* generators)
        ```
        """
        raise NotImplementedError("Set comprehension not implemented")

    def visit_DictComp(self, node: ast.DictComp) -> None:
        """
        辞書内包表記を処理する

        ```asdl
        DictComp(expr key, expr value, comprehension* generators)
        ```
        """
        raise NotImplementedError("Dict comprehension not implemented")

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        """
        ジェネレータ式を処理する

        ```asdl
        GeneratorExp(expr elt, comprehension* generators)
        ```
        """
        raise NotImplementedError("Generator expression not implemented")

    def visit_Compare(self, node: ast.Compare) -> ir.Value:
        if len(node.ops) != 1 or len(node.comparators) != 1:
            raise NotImplementedError("Only single comparison supported")
        lhs = self.require_value(node.left, self.visit(node.left))
        rhs = self.require_value(node.comparators[0], self.visit(node.comparators[0]))
        lhs, rhs = self.coerce_operands_for_binary(lhs, rhs, self._loc(node))
        op = node.ops[0]

        # In native mode, use arith.cmpi
        if self._in_native_func:
            return self._handle_primitive_compare(op, lhs, rhs, self._loc(node))

        # In object mode, use py.num.* comparisons
        bool_type = self.get_py_type("!py.bool")
        with self._loc(node), self.insertion_point():
            if isinstance(op, ast.LtE):
                return py_ops.NumLeOp(bool_type, lhs, rhs).result
            if isinstance(op, ast.Gt):
                return py_ops.NumGtOp(bool_type, lhs, rhs).result
            if isinstance(op, ast.Lt):
                return py_ops.NumLtOp(bool_type, lhs, rhs).result
            if isinstance(op, ast.GtE):
                return py_ops.NumGeOp(bool_type, lhs, rhs).result
            if isinstance(op, ast.Eq):
                return py_ops.NumEqOp(bool_type, lhs, rhs).result
            if isinstance(op, ast.NotEq):
                return py_ops.NumNeOp(bool_type, lhs, rhs).result
        raise NotImplementedError(
            "Only <, <=, >, >=, ==, != comparisons supported in object mode"
        )

    def _handle_primitive_compare(
        self, op: ast.cmpop, lhs: ir.Value, rhs: ir.Value, loc: ir.Location
    ) -> ir.Value:
        """Handle comparison on primitive types using arith.cmpi."""
        lhs_type = lhs.type
        if isinstance(lhs_type, ir.ShapedType):
            raise NotImplementedError("Tensor comparisons are not supported yet")
        is_float = self.is_primitive_float_type(lhs_type)

        # Map Python comparison operators to arith predicates
        # For integers (signed): slt, sle, sgt, sge, eq, ne
        # For floats: olt, ole, ogt, oge, oeq, one (ordered)
        with loc, self.insertion_point():
            if is_float:
                if isinstance(op, ast.Lt):
                    pred = arith_ops.CmpFPredicate.OLT
                elif isinstance(op, ast.LtE):
                    pred = arith_ops.CmpFPredicate.OLE
                elif isinstance(op, ast.Gt):
                    pred = arith_ops.CmpFPredicate.OGT
                elif isinstance(op, ast.GtE):
                    pred = arith_ops.CmpFPredicate.OGE
                elif isinstance(op, ast.Eq):
                    pred = arith_ops.CmpFPredicate.OEQ
                elif isinstance(op, ast.NotEq):
                    pred = arith_ops.CmpFPredicate.ONE
                else:
                    raise NotImplementedError(
                        f"Unsupported float comparison: {type(op).__name__}"
                    )
                return arith_ops.CmpFOp(pred, lhs, rhs).result
            else:
                if isinstance(op, ast.Lt):
                    pred = arith_ops.CmpIPredicate.slt
                elif isinstance(op, ast.LtE):
                    pred = arith_ops.CmpIPredicate.sle
                elif isinstance(op, ast.Gt):
                    pred = arith_ops.CmpIPredicate.sgt
                elif isinstance(op, ast.GtE):
                    pred = arith_ops.CmpIPredicate.sge
                elif isinstance(op, ast.Eq):
                    pred = arith_ops.CmpIPredicate.eq
                elif isinstance(op, ast.NotEq):
                    pred = arith_ops.CmpIPredicate.ne
                else:
                    raise NotImplementedError(
                        f"Unsupported integer comparison: {type(op).__name__}"
                    )
                return arith_ops.CmpIOp(pred, lhs, rhs).result

    def visit_Await(self, node: ast.Await) -> None:
        """
        await式を処理する

        ```asdl
        Await(expr value)
        ```
        """
        raise NotImplementedError("Await expression not implemented")

    def visit_Yield(self, node: ast.Yield) -> None:
        """
        yield式を処理する

        ```asdl
        Yield(expr? value)
        ```
        """
        raise NotImplementedError("Yield expression not implemented")

    def visit_YieldFrom(self, node: ast.YieldFrom) -> None:
        """
        yield from式を処理する

        ```asdl
        YieldFrom(expr value)
        ```
        """
        raise NotImplementedError("YieldFrom expression not implemented")

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
            if (
                returned_function_info is not None
                and str(result_type).startswith("!py.func<")
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
            exc_null = py_ops.ExceptionNullOp(
                self.get_py_type("!py.exception")
            ).result
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
        if (
            returned_function_info is not None
            and str(result_type).startswith("!py.func<")
        ):
            result = self._materialize_known_callable_result(
                result, returned_function_info, loc
            )
        return result

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

        # For now, we need to determine the class from the object type
        # This is a simplified implementation
        obj_type = obj.type
        obj_type_str = str(obj_type)

        # Extract class name from type like !py.class<"Counter">
        if obj_type_str.startswith('!py.class<"') and obj_type_str.endswith('">'):
            class_name = obj_type_str[len('!py.class<"') : -len('">')]  # noqa
        else:
            raise NotImplementedError(
                f"Method calls only supported on class instances, got {obj_type_str}"
            )

        class_info = self.lookup_class(class_name)
        if class_info is None:
            raise NameError(f"Unknown class '{class_name}'")

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

    def _prepare_list_element_for_storage(
        self, value: ir.Value, element_type: ir.Type, loc: ir.Location
    ) -> ir.Value:
        if value.type != element_type:
            value = self.coerce_value_to_type(value, element_type, loc)
        return value

    def _handle_prim_constructor(
        self, node: ast.Call, base_type: str, loc: ir.Location
    ) -> ir.Value:
        """
        Handle Int[N](value) or Float[N](value) constructor.

        Args:
            node: The Call AST node
            base_type: "Int" or "Float"
            loc: Source location
        """
        if len(node.args) != 1:
            raise ValueError(f"{base_type}[N]() requires exactly 1 argument")

        # Get the bit width from the subscript
        assert isinstance(node.func, ast.Subscript)
        if not isinstance(node.func.slice, ast.Constant) or not isinstance(
            node.func.slice.value, int
        ):
            raise ValueError(f"{base_type} requires an integer bit width")
        bits = node.func.slice.value

        # Get the MLIR type
        prim_type = self.get_primitive_type_from_spec(base_type, bits)

        value_node = node.args[0]

        # Handle constant values (compile-time conversion)
        if isinstance(value_node, ast.Constant):
            value = value_node.value

            with loc, self.insertion_point():
                if base_type == "Int" and isinstance(value, int):
                    attr = ir.IntegerAttr.get(prim_type, value)
                    result = arith_ops.ConstantOp(prim_type, attr).result
                    # Store the constant value for cross-region access
                    # (will be associated with variable name in visit_Assign)
                    self._pending_prim_const = (prim_type, value)
                    return result
                elif base_type == "Float" and isinstance(value, (int, float)):
                    attr = ir.FloatAttr.get(prim_type, float(value))
                    result = arith_ops.ConstantOp(prim_type, attr).result
                    self._pending_prim_const = (prim_type, float(value))
                    return result
                else:
                    raise ValueError(
                        f"Cannot convert {type(value).__name__} to {base_type}[{bits}]"
                    )

        # For non-constant values, generate py.cast.to_prim
        py_value = self.require_value(value_node, self.visit(value_node))
        with loc, self.insertion_point():
            return py_ops.CastToPrimOp(
                prim_type, py_value, ir.StringAttr.get("exact", self.ctx)
            ).result

    def _handle_from_prim(self, node: ast.Call, loc: ir.Location) -> ir.Value:
        """
        Handle from_prim(prim_value) call.

        Converts a primitive value back to a Python object.
        Generates py.cast.from_prim operation.
        """
        self._ensure_not_in_native("from_prim", loc)
        if len(node.args) != 1:
            raise ValueError("from_prim() requires exactly 1 argument")

        prim_value = self.require_value(node.args[0], self.visit(node.args[0]))

        # Determine the result Python type based on primitive type
        prim_type = prim_value.type
        prim_type_str = str(prim_type)

        if prim_type_str.startswith("i"):
            # Integer type -> !py.int
            result_type = self.get_py_type("!py.int")
        elif prim_type_str.startswith("f"):
            # Float type -> !py.float
            result_type = self.get_py_type("!py.float")
        elif isinstance(prim_type, ir.RankedTensorType):
            # Tensor type -> !py.str (repr carrier)
            result_type = self.get_py_type("!py.str")
        else:
            raise ValueError(
                f"Cannot convert primitive type {prim_type_str} to Python object"
            )

        with loc, self.insertion_point():
            return py_ops.CastFromPrimOp(result_type, prim_value).result

    def visit_FormattedValue(self, node: ast.FormattedValue) -> None:
        """
        フォーマット済み値を処理する

        ```asdl
        FormattedValue(expr value, int? conversion, expr? format_spec)
        ```
        """
        raise NotImplementedError("Formatted value not implemented")

    def visit_JoinedStr(self, node: ast.JoinedStr) -> None:
        """
        f文字列を処理する

        ```asdl
        JoinedStr(expr* values)
        ```
        """
        raise NotImplementedError("Joined string not implemented")

    def visit_Attribute(self, node: ast.Attribute) -> ir.Value | None:
        """
        属性アクセスを処理する

        ```asdl
        Attribute(expr value, identifier attr, expr_context ctx)
        ```
        """
        obj = self.require_value(node.value, self.visit(node.value))
        result_type = self.get_attribute_type(obj.type, node.attr)

        with self._loc(node), self.insertion_point():
            return py_ops.AttrGetOp(result_type, obj, node.attr).result

    def visit_Subscript(self, node: ast.Subscript) -> None:
        """
        添字アクセスを処理する

        ```asdl
        Subscript(expr value, expr slice, expr_context ctx)
        ```
        """
        container = self.require_value(node.value, self.visit(node.value))
        element_type = self.get_list_element_type(container.type)
        if element_type is not None:
            index = self.require_value(node.slice, self.visit(node.slice))
            with self._loc(node), self.insertion_point():
                return py_ops.ListGetOp(element_type, container, index).result

        dict_types = self.get_dict_key_value_types(container.type)
        if dict_types is not None:
            key_type, value_type = dict_types
            key = self.require_value(node.slice, self.visit(node.slice))
            key = self.coerce_value_to_type(key, key_type, self._loc(node))
            with self._loc(node), self.insertion_point():
                return py_ops.DictGetOp(value_type, container, key).result

        if element_type is None:
            raise NotImplementedError(
                f"Subscript access is only supported on !py.list or !py.dict values, got {container.type}"
            )

    def visit_Starred(self, node: ast.Starred) -> None:
        """
        a, b* = it のようなスター式を処理する

        ```asdl
        Starred(expr value, expr_context ctx)
        ```
        """
        raise NotImplementedError("Starred expression not implemented")

    def visit_List(self, node: ast.List) -> None:
        """
        リストリテラルを処理する。
        PyListObject*を生成し、要素を追加する。

        ```asdl
        List(expr* elts, expr_context ctx)
        ```
        """
        if not node.elts:
            raise NotImplementedError("Empty list literal is not supported yet")

        values = [self.require_value(elt, self.visit(elt)) for elt in node.elts]
        element_type = values[0].type
        list_type = self.get_py_type(f"!py.list<{element_type}>")

        with self._loc(node), self.insertion_point():
            result = py_ops.ListNewOp(list_type).result

        for value in values:
            prepared = self._prepare_list_element_for_storage(
                value, element_type, self._loc(node)
            )
            with self._loc(node), self.insertion_point():
                py_ops.ListAppendOp(result, prepared)
        return result

    def visit_Tuple(self, node: ast.Tuple) -> None:
        """
        タプルリテラル
        ```asdl
        Tuple(expr* elts, expr_context ctx)
        ```
        """
        raise NotImplementedError("Tuple literal not implemented")

    def visit_Slice(self, node: ast.Slice) -> None:
        """
        スライス式
        ```asdl
        Slice(expr? lower, expr? upper, expr? step)
        ```
        """
        raise NotImplementedError("Slice expression not implemented")
