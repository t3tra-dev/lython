from __future__ import annotations

import ast
from collections.abc import Mapping
from typing import TYPE_CHECKING

from ..mlir import ir
from ._base import BaseVisitor
from .expr_parts import (
    ExprAsyncCallMixin,
    ExprCallableCloneMixin,
    ExprCallableRemapMixin,
    ExprCallableSummaryMixin,
    ExprCallArgsMixin,
    ExprCallMethodsMixin,
    ExprCallMixin,
    ExprContainerMixin,
    ExprInvokeMixin,
    ExprLiteralMixin,
    ExprMiscMixin,
    ExprNameMixin,
    ExprNativeCallMixin,
    ExprOpsMixin,
)

if TYPE_CHECKING:
    from .contracts import VisitorRuntime

__all__ = ["ExprVisitor"]


class ExprVisitor(
    ExprAsyncCallMixin,
    ExprCallableCloneMixin,
    ExprCallableRemapMixin,
    ExprCallableSummaryMixin,
    ExprCallArgsMixin,
    ExprCallMethodsMixin,
    ExprCallMixin,
    ExprContainerMixin,
    ExprInvokeMixin,
    ExprLiteralMixin,
    ExprMiscMixin,
    ExprNameMixin,
    ExprNativeCallMixin,
    ExprOpsMixin,
    BaseVisitor,
):
    """
    式(expr)ノードの訪問を担当するクラス
    Python ソースの式を一対一で py dialect へ写像する。
    型解決済みの値は Lython の静的型に対応し、generic object 経路へ
    暗黙 materialize しない。

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
        subvisitors: Mapping[str, VisitorRuntime],
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
