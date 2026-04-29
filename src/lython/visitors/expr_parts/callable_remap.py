# pyright: reportAttributeAccessIssue=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false
from __future__ import annotations

from ...mlir import ir
from ...mlir.dialects import _lython_ops_gen as py_ops
from ...mlir.dialects import arith as arith_ops
from .._base import FunctionInfo


class ExprCallableRemapMixin:
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
                        py_ops.AttrGetOp(
                            value.type, remapped_object, attributes["name"]
                        ).result
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
                    result = py_ops.ListGetOp(
                        value.type, remapped_list, remapped_index
                    ).result
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
                    result = py_ops.NumEqOp(
                        value.type, remapped_lhs, remapped_rhs
                    ).result
                elif op_name == "py.num.ne":
                    result = py_ops.NumNeOp(
                        value.type, remapped_lhs, remapped_rhs
                    ).result
                elif op_name == "py.num.lt":
                    result = py_ops.NumLtOp(
                        value.type, remapped_lhs, remapped_rhs
                    ).result
                elif op_name == "py.num.le":
                    result = py_ops.NumLeOp(
                        value.type, remapped_lhs, remapped_rhs
                    ).result
                elif op_name == "py.num.gt":
                    result = py_ops.NumGtOp(
                        value.type, remapped_lhs, remapped_rhs
                    ).result
                else:
                    result = py_ops.NumGeOp(
                        value.type, remapped_lhs, remapped_rhs
                    ).result
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
