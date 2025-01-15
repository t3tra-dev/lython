from __future__ import annotations

import ast
from typing import Any

from ..ir import IRBuilder
from .base import BaseVisitor
from .expr import ExprVisitor

__all__ = ["StmtVisitor"]


class StmtVisitor(BaseVisitor):
    def __init__(self, builder: IRBuilder):
        self.builder = builder
        self.expr_visitor = ExprVisitor(builder)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """関数定義の処理"""
        # 関数名を取得
        func_name = node.name

        # 引数の処理
        arg_types = []
        arg_names = []
        for arg in node.args.args:
            arg_types.append("ptr")  # すべての引数をPyObject*として扱う
            arg_names.append(arg.arg)

        # 関数本体の開始
        self.builder.emit("")  # 空行を追加
        self.builder.emit("; Function definition")
        self.builder.emit(
            f"define dso_local ptr @{func_name}({', '.join(f'ptr noundef %{name}' for name in arg_names)}) #0 {{"
        )
        self.builder.emit("entry:")
        self.builder.emit("  ; Function body")

        # 関数本体の処理
        has_return = False
        for stmt in node.body:
            if isinstance(stmt, ast.Return):
                has_return = True
            self.visit(stmt)

        # 明示的なreturnがない場合のみNoneを返す
        if not has_return:
            self.builder.emit("  ret ptr @Py_None")
        self.builder.emit("}")

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        raise NotImplementedError("Async function definition not supported")

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        raise NotImplementedError("Class definition not supported")

    def visit_Return(self, node: ast.Return) -> None:
        """return文の処理"""
        if node.value is None:
            # return Noneの場合
            self.builder.emit("  ret ptr @Py_None")
        else:
            # 戻り値の評価
            value_ptr = self.expr_visitor.visit(node.value)
            self.builder.emit(f"  ret ptr {value_ptr}")

    def visit_Delete(self, node: ast.Delete) -> None:
        raise NotImplementedError("Delete statement not supported")

    def visit_Assign(self, node: ast.Assign) -> None:
        raise NotImplementedError("Assignment not supported")

    def visit_Expr(self, node: ast.Expr) -> Any:
        """式文の処理"""
        return self.expr_visitor.visit(node.value)

    # 他のstmt関連のvisitメソッドも同様に実装

    def visit_If(self, node: ast.If) -> None:
        """if文の処理"""
        # 条件式の評価結果を取得
        cond_ptr = self.expr_visitor.visit(node.test)

        # 一意なラベル名を生成
        then_label = f"if.then.{self.builder.get_label_counter()}"
        else_label = f"if.else.{self.builder.get_label_counter()}"
        end_label = f"if.end.{self.builder.get_label_counter()}"

        # PyIntObjectのvalueフィールドへのポインタを取得
        cond_val_ptr = self.builder.get_temp_name()
        self.builder.emit(
            f"  {cond_val_ptr} = getelementptr %struct.PyIntObject, ptr {cond_ptr}, i32 0, i32 1"
        )

        # 条件値をロード
        cond_val = self.builder.get_temp_name()
        self.builder.emit(f"  {cond_val} = load i64, ptr {cond_val_ptr}")

        # i64をi1に変換
        cond_bool = self.builder.get_temp_name()
        self.builder.emit(f"  {cond_bool} = trunc i64 {cond_val} to i1")

        # 条件分岐
        self.builder.emit(f"  br i1 {cond_bool}, label %{then_label}, label %{else_label}")

        # then部分の処理
        self.builder.emit(f"{then_label}:")
        for stmt in node.body:
            self.visit(stmt)
        self.builder.emit(f"  br label %{end_label}")

        # else部分の処理
        self.builder.emit(f"{else_label}:")
        for stmt in node.orelse:
            self.visit(stmt)
        self.builder.emit(f"  br label %{end_label}")

        # 終了ラベル
        self.builder.emit(f"{end_label}:")
