from __future__ import annotations

import ast

from ..mlir import ir
from ..mlir.dialects import _lython_ops_gen as py_ops
from ..mlir.dialects import arith as arith_ops
from ._base import BaseVisitor
from .expr_parts import (
    ExprCallableCloneMixin,
    ExprCallableRemapMixin,
    ExprCallableSummaryMixin,
    ExprCallArgsMixin,
    ExprCallMethodsMixin,
    ExprCallMixin,
    ExprContainerMixin,
    ExprLiteralMixin,
    ExprMiscMixin,
    ExprNameMixin,
    ExprOpsMixin,
)

__all__ = ["ExprVisitor"]


class ExprVisitor(
    ExprCallableCloneMixin,
    ExprCallableRemapMixin,
    ExprCallableSummaryMixin,
    ExprCallArgsMixin,
    ExprCallMethodsMixin,
    ExprCallMixin,
    ExprContainerMixin,
    ExprLiteralMixin,
    ExprMiscMixin,
    ExprNameMixin,
    ExprOpsMixin,
    BaseVisitor,
):
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
