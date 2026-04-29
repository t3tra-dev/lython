# pyright: reportAttributeAccessIssue=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false
from __future__ import annotations

import ast

from ...mlir import ir
from ...mlir.dialects import _lython_ops_gen as py_ops
from ...mlir.dialects import arith as arith_ops
from ...mlir.dialects import tensor as tensor_ops


class ExprOpsMixin:
    """Expression lowering for unary, binary, and comparison operations."""

    def visit_BinOp(self, node: ast.BinOp) -> ir.Value:
        lhs = self.require_value(node.left, self.visit(node.left))
        rhs = self.require_value(node.right, self.visit(node.right))
        lhs, rhs = self.coerce_operands_for_binary(lhs, rhs, self._loc(node))

        if self._in_native_func or (
            not self.is_py_type(lhs.type) and not self.is_py_type(rhs.type)
        ):
            return self._handle_primitive_binop(node.op, lhs, rhs, self._loc(node))

        with self._loc(node), self.insertion_point():
            if isinstance(node.op, ast.Add):
                return py_ops.NumAddOp(lhs, rhs).result
            if isinstance(node.op, ast.Sub):
                return py_ops.NumSubOp(lhs, rhs).result
        raise NotImplementedError("Unsupported binary operation")

    def _handle_primitive_binop(
        self, op: ast.operator, lhs: ir.Value, rhs: ir.Value, loc: ir.Location
    ) -> ir.Value:
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
            if isinstance(op, ast.Sub):
                if is_float:
                    return arith_ops.SubFOp(lhs, rhs).result
                if is_int:
                    return arith_ops.SubIOp(lhs, rhs).result
                raise NotImplementedError("Unsupported primitive sub type")
            if isinstance(op, ast.Mult):
                if is_float:
                    return arith_ops.MulFOp(lhs, rhs).result
                if is_int:
                    return arith_ops.MulIOp(lhs, rhs).result
                raise NotImplementedError("Unsupported primitive mul type")
            if isinstance(op, ast.FloorDiv):
                if is_float:
                    raise NotImplementedError("Floor division on floats not supported")
                if is_int:
                    return arith_ops.DivSIOp(lhs, rhs).result
                raise NotImplementedError("Unsupported primitive div type")
            if isinstance(op, ast.Div):
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
            if isinstance(op, ast.MatMult):
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
            if isinstance(op, ast.Mod):
                if is_float:
                    return arith_ops.RemFOp(lhs, rhs).result
                if is_int:
                    return arith_ops.RemSIOp(lhs, rhs).result
                raise NotImplementedError("Unsupported primitive rem type")
            raise NotImplementedError(
                f"Unsupported primitive binary operation: {type(op).__name__}"
            )

    def visit_UnaryOp(self, node: ast.UnaryOp) -> ir.Value:
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
            raise NotImplementedError(f"Unary plus is not supported for {operand_type}")

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

    def visit_Compare(self, node: ast.Compare) -> ir.Value:
        if len(node.ops) != 1 or len(node.comparators) != 1:
            raise NotImplementedError("Only single comparison supported")
        lhs = self.require_value(node.left, self.visit(node.left))
        rhs = self.require_value(node.comparators[0], self.visit(node.comparators[0]))
        lhs, rhs = self.coerce_operands_for_binary(lhs, rhs, self._loc(node))
        op = node.ops[0]

        if self._in_native_func:
            return self._handle_primitive_compare(op, lhs, rhs, self._loc(node))

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
        lhs_type = lhs.type
        if isinstance(lhs_type, ir.ShapedType):
            raise NotImplementedError("Tensor comparisons are not supported yet")
        is_float = self.is_primitive_float_type(lhs_type)

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
