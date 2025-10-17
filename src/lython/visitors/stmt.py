from __future__ import annotations

import ast
from typing import Any

from ..mlir import ir
from ._base import BaseVisitor

__all__ = ["StmtVisitor"]


class StmtVisitor(BaseVisitor):
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

    def __init__(self, ctx: ir.Context):
        super().__init__(ctx)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """
        関数定義:
        def hoge(n: int) -> int:
            ...
        などを受け取り、静的型のIRを生成する

        ```asdl
        FunctionDef(
            identifier name,
            arguments args,
            stmt* body,
            expr* decorator_list,
            expr? returns,
            string? type_comment,
            type_param* type_params
        )
        ```
        """
        raise NotImplementedError("Function definition not supported")

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """
        非同期関数の定義を処理する

        ```asdl
        AsyncFunctionDef(
            identifier name,
            arguments args,
            stmt* body,
            expr* decorator_list,
            expr? returns,
            string? type_comment,
            type_param* type_params
        )
        ```
        """
        raise NotImplementedError("Async function definition not supported")

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """
        クラス定義

        ```asdl
        ClassDef(
            identifier name,
            expr* bases,
            keyword* keywords,
            stmt* body,
            expr* decorator_list,
            type_param* type_params
        )
        ```
        """
        raise NotImplementedError("Class definition not supported")

    def visit_Return(self, node: ast.Return) -> None:
        """
        関数の返り値を定めるreturn文を処理する

        ```asdl
        Return(expr? value)
        ```
        """
        raise NotImplementedError("Return statement not supported")

    def visit_Delete(self, node: ast.Delete) -> None:
        """
        変数を削除するdelete文を処理する

        ```asdl
        Delete(expr* targets)
        ```
        """
        raise NotImplementedError("Delete statement not supported")

    def visit_Assign(self, node: ast.Assign) -> None:
        """
        代入演算子 = を処理する

        ```asdl
        Assign(expr* targets, expr value, string? type_comment)
        ```
        """
        raise NotImplementedError("Assignment statement not supported")

    def visit_TypeAlias(self, node: ast.TypeAlias) -> None:
        """
        型エイリアスを処理する

        ```asdl
        TypeAlias(
            expr name,
            type_param* type_params,
            expr value
        )
        """
        raise NotImplementedError("Type alias statement not implemented")

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        """
        a += 1 のような累積代入を処理する

        ```asdl
        AugAssign(
            expr target,
            operator op,
            expr value
        )
        """
        raise NotImplementedError("Augmented assignment statement not implemented")

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        """
        c: int のような型注釈を持つ代入を処理する

        ```asdl
        -- 'simple' indicates that we annotate simple name without parens
        AnnAssign(
            expr target,
            expr annotation,
            expr? value,
            int simple
        )
        ```
        """
        raise NotImplementedError(
            "An assignment with a type annotation is not implemented"
        )

    def visit_For(self, node: ast.For) -> None:
        """
        for文の処理をする

        ```asdl
        For(
            expr target,
            expr iter,
            stmt* body,
            stmt* orelse,
            string? type_comment
        )
        ```
        """
        raise NotImplementedError("For statement not implemented")

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        """
        非同期for文の処理をする

        ```asdl
        AsyncFor(
            expr target,
            expr iter,
            stmt* body,
            stmt* orelse,
            string? type_comment
        )
        ```
        """
        raise NotImplementedError("Async for statement not implemented")

    def visit_While(self, node: ast.While) -> None:
        """
        while文を処理する

        ```asdl
        While(
            expr test,
            stmt* body,
            stmt* orelse
        )
        ```
        """
        raise NotImplementedError("While statement not implemented")

    def visit_If(self, node: ast.If) -> None:
        """
        if文:
          if test:
              ...
          else:
              ...

        ```asdl
        If(expr test, stmt* body, stmt* orelse)
        ```
        """
        raise NotImplementedError("If statement not implemented")

    def visit_With(self, node: ast.With) -> None:
        """
        with文を処理する

        ```asdl
        With(withitem* items, stmt* body, string? type_comment)
        ```
        """
        raise NotImplementedError("With statement not implemented")

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
        """
        非同期with文を処理する

        ```asdl
        AsyncWith(withitem* items, stmt* body, string? type_comment)
        ```
        """
        raise NotImplementedError("Async with statement not implemented")

    def visit_Match(self, node: ast.Match) -> None:
        """
        match文を処理する

        ```asdl
        Match(expr subject, match_case* cases)
        ```
        """
        raise NotImplementedError("Match statement not implemented")

    def visit_Raise(self, node: ast.Raise) -> None:
        """
        raise文を処理する

        ```asdl
        Raise(expr? exc, expr? cause)
        ```
        """
        raise NotImplementedError("Raise statement not implemented")

    def visit_Try(self, node: ast.Try) -> None:
        """
        Try文を処理する

        ```asdl
        Try(
            stmt* body,
            excepthandler* handlers,
            stmt* orelse,
            stmt* finalbody
        )
        ```
        """
        raise NotImplementedError("Try statement not implemented")

    def visit_TryStar(self, node: ast.TryStar) -> None:
        """
        except*節が続くtryブロックを処理する

        ```asdl
        TryStar(
            stmt* body,
            excepthandler* handlers,
            stmt* orelse,
            stmt* finalbody
        )
        ```
        """
        raise NotImplementedError("Try star statement not implemented")

    def visit_Assert(self, node: ast.Assert) -> None:
        """
        assert文を処理する

        ```asdl
        Assert(expr test, expr? msg)
        ```
        """
        raise NotImplementedError("Assert statement not implemented")

    def visit_Import(self, node: ast.Import) -> None:
        """
        import文を処理する

        ```asdl
        Import(alias* names)
        ```
        """
        raise NotImplementedError("Import statement not implemented")

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """
        from ... import文を処理する

        ```asdl
        ImportFrom(identifier? module, alias* names, int? level)
        ```
        """
        raise NotImplementedError("Import from statement not implemented")

    def visit_Global(self, node: ast.Global) -> None:
        """
        global文を処理する

        ```asdl
        Global(identifier* names)
        ```
        """
        raise NotImplementedError("Global statement not implemented")

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        """
        nonlocal文を処理する

        ```asdl
        Nonlocal(identifier* names)
        ```
        """
        raise NotImplementedError("Nonlocal statement not implemented")

    def visit_Expr(self, node: ast.Expr) -> Any:
        """
        式文
        戻り値は破棄してよいので、一応計算はするが特に変数には入れない

        ```asdl
        Expr(expr value)
        ```
        """
        raise NotImplementedError("Expression statement not implemented")

    def visit_Pass(self, node: ast.Pass) -> None:
        """
        pass文を処理する

        ```asdl
        Pass
        ```
        """
        raise NotImplementedError("Pass statement not implemented")

    def visit_Break(self, node: ast.Break) -> None:
        """
        break文を処理する

        ```asdl
        Break
        ```
        """
        raise NotImplementedError("Break statement not implemented")

    def visit_Continue(self, node: ast.Continue) -> None:
        """
        continue文を処理する

        ```asdl
        Continue
        ```
        """
        raise NotImplementedError("Continue statement not implemented")
