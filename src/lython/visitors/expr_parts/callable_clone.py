# pyright: reportAttributeAccessIssue=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false
from __future__ import annotations

from ...mlir import ir
from ...mlir.dialects import _lython_ops_gen as py_ops
from ...mlir.dialects import arith as arith_ops


class ExprCallableCloneMixin:
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
            if op_name == "py.publish":
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
                py_ops.DictInsertOp(base, key, item)
                return base
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
                segment_sizes = [
                    int(size) for size in attributes["operandSegmentSizes"]
                ]
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
            if op_name == "py.repr":
                cloned_input = self._clone_bound_callable_metadata(operands[0], loc)
                return py_ops.ReprOp(value.type, cloned_input).result
            if op_name == "py.add":
                lhs = self._clone_bound_callable_metadata(operands[0], loc)
                rhs = self._clone_bound_callable_metadata(operands[1], loc)
                return py_ops.AddOp(value.type, lhs, rhs).result
            if op_name == "py.sub":
                lhs = self._clone_bound_callable_metadata(operands[0], loc)
                rhs = self._clone_bound_callable_metadata(operands[1], loc)
                return py_ops.SubOp(value.type, lhs, rhs).result
            if op_name == "py.eq":
                lhs = self._clone_bound_callable_metadata(operands[0], loc)
                rhs = self._clone_bound_callable_metadata(operands[1], loc)
                return py_ops.EqOp(value.type, lhs, rhs).result
            if op_name == "py.ne":
                lhs = self._clone_bound_callable_metadata(operands[0], loc)
                rhs = self._clone_bound_callable_metadata(operands[1], loc)
                return py_ops.NeOp(value.type, lhs, rhs).result
            if op_name == "py.lt":
                lhs = self._clone_bound_callable_metadata(operands[0], loc)
                rhs = self._clone_bound_callable_metadata(operands[1], loc)
                return py_ops.LtOp(value.type, lhs, rhs).result
            if op_name == "py.le":
                lhs = self._clone_bound_callable_metadata(operands[0], loc)
                rhs = self._clone_bound_callable_metadata(operands[1], loc)
                return py_ops.LeOp(value.type, lhs, rhs).result
            if op_name == "py.gt":
                lhs = self._clone_bound_callable_metadata(operands[0], loc)
                rhs = self._clone_bound_callable_metadata(operands[1], loc)
                return py_ops.GtOp(value.type, lhs, rhs).result
            if op_name == "py.ge":
                lhs = self._clone_bound_callable_metadata(operands[0], loc)
                rhs = self._clone_bound_callable_metadata(operands[1], loc)
                return py_ops.GeOp(value.type, lhs, rhs).result
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
