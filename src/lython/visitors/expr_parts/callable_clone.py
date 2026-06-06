from __future__ import annotations

from typing import TYPE_CHECKING

from ...mlir import ir
from ...mlir.dialects import _lython_ops_gen as py_ops
from ...mlir.dialects import arith as arith_ops
from ..mlir_access import (
    block_owner,
    op_attributes,
    op_name,
    op_operand_segment_sizes,
    op_operands,
    owner_parent,
    value_operation,
    value_owner,
)

if TYPE_CHECKING:
    from ..contracts import VisitorRuntime
else:
    VisitorRuntime = object


class ExprCallableCloneMixin(VisitorRuntime):
    def _clone_bound_callable_metadata(
        self, value: ir.Value, loc: ir.Location
    ) -> ir.Value:
        current_parent = block_owner(self.current_block)
        owner = value_owner(value)
        if current_parent is not None and owner is not None:
            parent = owner_parent(owner)
            if parent is not None and parent == current_parent:
                return value
        op = value_operation(value)
        if op is None:
            return value
        name = op_name(op)
        operands = op_operands(op)
        attributes = op_attributes(op)

        with loc, self.insertion_point():
            if name == "py.publish":
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
            if name == "py.str.constant":
                return py_ops.StrConstantOp(value.type, attributes["value"]).result
            if name == "py.int.constant":
                return py_ops.IntConstantOp(value.type, attributes["value"]).result
            if name == "py.float.constant":
                return py_ops.FloatConstantOp(value.type, attributes["value"]).result
            if name == "arith.constant":
                return arith_ops.ConstantOp(value.type, attributes["value"]).result
            if name == "py.none":
                return py_ops.NoneOp(value.type).result
            if name == "py.tuple.empty":
                return py_ops.TupleEmptyOp(value.type).result
            if name == "py.tuple.create":
                elements = [
                    self._clone_bound_callable_metadata(element, loc)
                    for element in operands
                ]
                return py_ops.TupleCreateOp(value.type, elements).result
            if name == "py.dict.empty":
                return py_ops.DictEmptyOp(value.type).result
            if name == "py.dict.insert":
                base = self._clone_bound_callable_metadata(operands[0], loc)
                key = self._clone_bound_callable_metadata(operands[1], loc)
                item = self._clone_bound_callable_metadata(operands[2], loc)
                py_ops.DictInsertOp(base, key, item)
                return base
            if name == "py.attr.get":
                cloned_object = self._clone_bound_callable_metadata(operands[0], loc)
                return py_ops.AttrGetOp(
                    value.type,
                    cloned_object,
                    attributes["name"],
                ).result
            if name == "py.list.get":
                cloned_list = self._clone_bound_callable_metadata(operands[0], loc)
                cloned_index = self._clone_bound_callable_metadata(operands[1], loc)
                return py_ops.ListGetOp(value.type, cloned_list, cloned_index).result
            if name == "py.func.object":
                return py_ops.FuncObjectOp(value.type, attributes["target"]).result
            if name == "py.make_function":
                segment_sizes = op_operand_segment_sizes(op)
                if not segment_sizes:
                    return value
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
            if name == "py.cast.to_prim":
                cloned_input = self._clone_bound_callable_metadata(operands[0], loc)
                return py_ops.CastToPrimOp(
                    value.type,
                    cloned_input,
                    attributes["mode"],
                ).result
            if name == "py.cast.from_prim":
                cloned_input = self._clone_bound_callable_metadata(operands[0], loc)
                return py_ops.CastFromPrimOp(value.type, cloned_input).result
            if name == "py.repr":
                cloned_input = self._clone_bound_callable_metadata(operands[0], loc)
                return py_ops.ReprOp(value.type, cloned_input).result
            if name == "py.add":
                lhs = self._clone_bound_callable_metadata(operands[0], loc)
                rhs = self._clone_bound_callable_metadata(operands[1], loc)
                return py_ops.AddOp(value.type, lhs, rhs).result
            if name == "py.sub":
                lhs = self._clone_bound_callable_metadata(operands[0], loc)
                rhs = self._clone_bound_callable_metadata(operands[1], loc)
                return py_ops.SubOp(value.type, lhs, rhs).result
            if name == "py.eq":
                lhs = self._clone_bound_callable_metadata(operands[0], loc)
                rhs = self._clone_bound_callable_metadata(operands[1], loc)
                return py_ops.EqOp(value.type, lhs, rhs).result
            if name == "py.ne":
                lhs = self._clone_bound_callable_metadata(operands[0], loc)
                rhs = self._clone_bound_callable_metadata(operands[1], loc)
                return py_ops.NeOp(value.type, lhs, rhs).result
            if name == "py.lt":
                lhs = self._clone_bound_callable_metadata(operands[0], loc)
                rhs = self._clone_bound_callable_metadata(operands[1], loc)
                return py_ops.LtOp(value.type, lhs, rhs).result
            if name == "py.le":
                lhs = self._clone_bound_callable_metadata(operands[0], loc)
                rhs = self._clone_bound_callable_metadata(operands[1], loc)
                return py_ops.LeOp(value.type, lhs, rhs).result
            if name == "py.gt":
                lhs = self._clone_bound_callable_metadata(operands[0], loc)
                rhs = self._clone_bound_callable_metadata(operands[1], loc)
                return py_ops.GtOp(value.type, lhs, rhs).result
            if name == "py.ge":
                lhs = self._clone_bound_callable_metadata(operands[0], loc)
                rhs = self._clone_bound_callable_metadata(operands[1], loc)
                return py_ops.GeOp(value.type, lhs, rhs).result
            if name == "arith.cmpi":
                lhs = self._clone_bound_callable_metadata(operands[0], loc)
                rhs = self._clone_bound_callable_metadata(operands[1], loc)
                return arith_ops.CmpIOp(attributes["predicate"], lhs, rhs).result
            if name == "arith.cmpf":
                lhs = self._clone_bound_callable_metadata(operands[0], loc)
                rhs = self._clone_bound_callable_metadata(operands[1], loc)
                return arith_ops.CmpFOp(attributes["predicate"], lhs, rhs).result
            if name == "arith.addi":
                lhs = self._clone_bound_callable_metadata(operands[0], loc)
                rhs = self._clone_bound_callable_metadata(operands[1], loc)
                return arith_ops.AddIOp(lhs, rhs).result
            if name == "arith.subi":
                lhs = self._clone_bound_callable_metadata(operands[0], loc)
                rhs = self._clone_bound_callable_metadata(operands[1], loc)
                return arith_ops.SubIOp(lhs, rhs).result
            if name == "arith.muli":
                lhs = self._clone_bound_callable_metadata(operands[0], loc)
                rhs = self._clone_bound_callable_metadata(operands[1], loc)
                return arith_ops.MulIOp(lhs, rhs).result
            if name == "arith.divsi":
                lhs = self._clone_bound_callable_metadata(operands[0], loc)
                rhs = self._clone_bound_callable_metadata(operands[1], loc)
                return arith_ops.DivSIOp(lhs, rhs).result
            if name == "arith.remsi":
                lhs = self._clone_bound_callable_metadata(operands[0], loc)
                rhs = self._clone_bound_callable_metadata(operands[1], loc)
                return arith_ops.RemSIOp(lhs, rhs).result
            if name == "arith.addf":
                lhs = self._clone_bound_callable_metadata(operands[0], loc)
                rhs = self._clone_bound_callable_metadata(operands[1], loc)
                return arith_ops.AddFOp(lhs, rhs).result
            if name == "arith.subf":
                lhs = self._clone_bound_callable_metadata(operands[0], loc)
                rhs = self._clone_bound_callable_metadata(operands[1], loc)
                return arith_ops.SubFOp(lhs, rhs).result
            if name == "arith.mulf":
                lhs = self._clone_bound_callable_metadata(operands[0], loc)
                rhs = self._clone_bound_callable_metadata(operands[1], loc)
                return arith_ops.MulFOp(lhs, rhs).result
            if name == "arith.divf":
                lhs = self._clone_bound_callable_metadata(operands[0], loc)
                rhs = self._clone_bound_callable_metadata(operands[1], loc)
                return arith_ops.DivFOp(lhs, rhs).result
            if name == "arith.remf":
                lhs = self._clone_bound_callable_metadata(operands[0], loc)
                rhs = self._clone_bound_callable_metadata(operands[1], loc)
                return arith_ops.RemFOp(lhs, rhs).result

        raise NotImplementedError(
            f"Cloning bound callable metadata from {name or type(op).__name__} is not supported yet"
        )
