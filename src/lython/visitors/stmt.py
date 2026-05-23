from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from ..mlir import ir
from ._base import BaseVisitor
from .stmt_parts import (
    StmtAssignMixin,
    StmtClassMixin,
    StmtControlMixin,
    StmtExceptionMixin,
    StmtFunctionMixin,
    StmtImportMixin,
    StmtMiscMixin,
)

if TYPE_CHECKING:
    from .contracts import VisitorRuntime

__all__ = ["StmtVisitor"]


class StmtVisitor(
    StmtAssignMixin,
    StmtClassMixin,
    StmtControlMixin,
    StmtExceptionMixin,
    StmtFunctionMixin,
    StmtImportMixin,
    StmtMiscMixin,
    BaseVisitor,
):
    """
    文(stmt)ノードを訪問し、
    MLIRを生成するクラス。

    ```asdl
    stmt = FunctionDef(identifier name, arguments args,
                       stmt* body, expr* decorator_list, expr? returns,
                       string? type_comment, type_param* type_params)
          | AsyncFunctionDef(identifier name, arguments args,
                             stmt* body, expr* decorator_list, expr? returns,
                             string? type_comment, type_param* type_params)

          | ClassDef(identifier name,
             expr* bases,
             keyword* keywords,
             stmt* body,
             expr* decorator_list,
             type_param* type_params)
          | Return(expr? value)

          | Delete(expr* targets)
          | Assign(expr* targets, expr value, string? type_comment)
          | TypeAlias(expr name, type_param* type_params, expr value)
          | AugAssign(expr target, operator op, expr value)
          -- 'simple' indicates that we annotate simple name without parens
          | AnnAssign(expr target, expr annotation, expr? value, int simple)

          -- use 'orelse' because else is a keyword in target languages
          | For(expr target, expr iter, stmt* body, stmt* orelse, string? type_comment)
          | AsyncFor(expr target, expr iter, stmt* body, stmt* orelse, string? type_comment)
          | While(expr test, stmt* body, stmt* orelse)
          | If(expr test, stmt* body, stmt* orelse)
          | With(withitem* items, stmt* body, string? type_comment)
          | AsyncWith(withitem* items, stmt* body, string? type_comment)

          | Match(expr subject, match_case* cases)

          | Raise(expr? exc, expr? cause)
          | Try(stmt* body, excepthandler* handlers, stmt* orelse, stmt* finalbody)
          | TryStar(stmt* body, excepthandler* handlers, stmt* orelse, stmt* finalbody)
          | Assert(expr test, expr? msg)

          | Import(alias* names)
          | ImportFrom(identifier? module, alias* names, int? level)

          | Global(identifier* names)
          | Nonlocal(identifier* names)
          | Expr(expr value)
          | Pass | Break | Continue

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
