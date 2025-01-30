from __future__ import annotations

import ast
from typing import Any

from ..ir import IRBuilder
from .base import BaseVisitor
from .expr import ExprVisitor

__all__ = ["StmtVisitor"]


class StmtVisitor(BaseVisitor):
    """
    文(stmt)ノードを訪問しIRを生成する
    静的型付けとして扱うためFunctionDefなどは引数型/戻り値型を注釈から参照し
    IR上のシグネチャを決定する(のが理想だが、ここでは最低限の実装に留める)

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
    """

    def __init__(self, builder: IRBuilder):
        self.builder = builder
        self.expr_visitor = ExprVisitor(builder)

    # ---------------------------
    # visit_FunctionDef
    # ---------------------------
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """
        関数定義:
          def hoge(n: int) -> int:
              ...
        などを受け取り、静的型のIRを生成する
        """
        func_name = node.name

        # 引数の型注釈読み取り (最低限)
        arg_types = []
        arg_names = []
        for arg in node.args.args:
            # ここではすべて i32 と仮定
            # TODO: annotation を見て分岐
            arg_types.append("i32")
            arg_names.append(arg.arg)

        # TODO: 戻り値型も annotation を見て i32 or ptr などにする
        # 一旦 i32 と仮定
        return_type = "i32"

        # IR出力
        self.builder.emit("")
        self.builder.emit(f"; Function definition: {func_name}")

        # 引数部
        joined_args = ", ".join(f"{t} %{name}" for t, name in zip(arg_types, arg_names))
        self.builder.emit(f"define {return_type} @{func_name}({joined_args}) #0 {{")
        self.builder.emit("entry:")

        # ステートメント群を処理
        self.visit_function_body(node.body, return_type)

        self.builder.emit("}")

    def visit_function_body(self, stmts: list[ast.stmt], return_type: str) -> None:
        """
        関数本体のステートメントを順に訪問し
        最後にreturnがなければデフォルトの返却を生成する
        """
        has_return = False
        for s in stmts:
            if isinstance(s, ast.Return):
                has_return = True
            self.visit(s)

        if not has_return:
            if return_type == "i32":
                # return 0
                self.builder.emit("  ret i32 0")
            elif return_type == "ptr":
                # return null
                self.builder.emit("  ret ptr null")
            else:
                self.builder.emit("  ret void")

    # ---------------------------
    # visit_Return
    # ---------------------------
    def visit_Return(self, node: ast.Return) -> None:
        """
        return文を処理
        戻り値が無ければデフォルト値
        """
        if node.value is None:
            # i32の場合 0を返す と仮定
            self.builder.emit("  ret i32 0")
        else:
            val = self.expr_visitor.visit(node.value)
            # 一旦 i32 と仮定
            self.builder.emit(f"  ret i32 {val}")

    # ---------------------------
    # visit_If
    # ---------------------------
    def visit_If(self, node: ast.If) -> None:
        """
        if文:
          if test:
              ...
          else:
              ...
        静的に i32(0以外) をtrueとみなすなど
        """
        cond_val = self.expr_visitor.visit(node.test)  # i32 0/1
        # i32 -> i1
        tmp_bool = self.builder.get_temp_name()
        self.builder.emit(f"  {tmp_bool} = icmp ne i32 {cond_val}, 0")

        then_label = f"if.then.{self.builder.get_label_counter()}"
        else_label = f"if.else.{self.builder.get_label_counter()}"
        end_label = f"if.end.{self.builder.get_label_counter()}"

        # 分岐
        self.builder.emit(f"  br i1 {tmp_bool}, label %{then_label}, label %{else_label}")

        # then
        self.builder.emit(f"{then_label}:")
        for s in node.body:
            self.visit(s)
        self.builder.emit(f"  br label %{end_label}")

        # else
        self.builder.emit(f"{else_label}:")
        for s in node.orelse:
            self.visit(s)
        self.builder.emit(f"  br label %{end_label}")

        # end
        self.builder.emit(f"{end_label}:")

    # ---------------------------
    # その他stmt
    # ---------------------------
    def visit_Expr(self, node: ast.Expr) -> Any:
        """
        式文
        戻り値は破棄してよいので、一応計算はするが特に変数には入れない
        """
        _ = self.expr_visitor.visit(node.value)
        # discard
        return None

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        raise NotImplementedError("Class definition not supported in static mode")

    def visit_Assign(self, node: ast.Assign) -> None:
        raise NotImplementedError("Assignment not supported in static mode")

    def visit_Delete(self, node: ast.Delete) -> None:
        raise NotImplementedError("Delete statement not supported in static mode")

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        raise NotImplementedError("Async function definition not supported")

    def visit_For(self, node: ast.For) -> None:
        raise NotImplementedError("For statement not implemented")

    def visit_While(self, node: ast.While) -> None:
        raise NotImplementedError("While statement not implemented")

    def visit_Try(self, node: ast.Try) -> None:
        raise NotImplementedError("Try statement not implemented")

    # など他のstmtも未実装の場合は同様に NotImplementedError

    def generic_visit(self, node: ast.AST) -> None:
        return super().generic_visit(node)
