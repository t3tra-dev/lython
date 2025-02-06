from __future__ import annotations

import ast
from typing import Any, List

from ..ir import IRBuilder
from .base import BaseVisitor, TypedValue
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
        super().__init__(builder)
        self.expr_visitor = ExprVisitor(builder)
        self.current_return_type = "i32"

        # ExprVisitorとシンボルテーブルを共有
        self.expr_visitor.symbol_table = self.symbol_table
        self.expr_visitor.function_signatures = self.function_signatures

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """
        関数定義:
          def hoge(n: int) -> int:
              ...
        などを受け取り、静的型のIRを生成する
        """
        func_name = node.name

        # 引数の型を解析
        arg_types = []

        for arg in node.args.args:
            arg_type = "i32"  # デフォルト値

            if arg.annotation:
                if isinstance(arg.annotation, ast.Name):
                    if arg.annotation.id == "int":
                        arg_type = "i32"
                    elif arg.annotation.id == "str":
                        arg_type = "ptr"
                    else:
                        raise NotImplementedError(f"Unsupported annotation: {arg.annotation.id}")
                else:
                    raise NotImplementedError("Complex annotations not supported")

            # シンボルテーブルに引数を登録
            self.set_symbol_type(arg.arg, arg_type)
            arg_types.append(arg_type)

        # 戻り値型を解析
        if node.returns is not None and isinstance(node.returns, ast.Name):
            if node.returns.id == "int":
                return_type = "i32"
            elif node.returns.id == "str":
                return_type = "ptr"
            else:
                raise NotImplementedError(f"Unsupported return annotation: {node.returns.id}")
        else:
            # fallback
            return_type = "i32"  # or raise error

        # この関数のシグネチャを登録
        self.set_function_signature(func_name, arg_types, return_type)

        # 現在の関数の戻り値型を更新
        self.current_return_type = return_type

        # IR出力
        self.builder.emit("")
        self.builder.emit(f"; Function definition: {func_name}")

        # LLVM IR の仮引数部
        joined_args = ", ".join(
            f"{t} %{arg.arg}" for t, arg in zip(arg_types, node.args.args)
        )
        self.builder.emit(f"define {return_type} @{func_name}({joined_args}) #0 {{")
        self.builder.emit("entry:")

        # 関数ボディの visit
        self.visit_function_body(node.body, return_type)

        self.builder.emit("}")

    def visit_function_body(self, stmts: List[ast.stmt], return_type: str) -> None:
        has_return = False
        for s in stmts:
            if isinstance(s, ast.Return):
                has_return = True
            self.visit(s)

        if not has_return:
            if return_type == "i32":
                self.builder.emit("  ret i32 0")
            elif return_type == "ptr":
                self.builder.emit("  ret ptr null")
            else:
                self.builder.emit("  ret void")

    def visit_Return(self, node: ast.Return) -> None:
        if node.value is None:
            # デフォルトの戻り値
            if self.current_return_type == "i32":
                self.builder.emit("  ret i32 0")
            elif self.current_return_type == "ptr":
                self.builder.emit("  ret ptr null")
            else:
                self.builder.emit("  ret void")
        else:
            val_tv = self.expr_visitor.visit(node.value)  # => TypedValue
            # 型チェック
            if val_tv.type_ != self.current_return_type:
                raise TypeError(f"Return type mismatch, expected {self.current_return_type}, got {val_tv.type_}")
            self.builder.emit(f"  ret {val_tv.type_} {val_tv.llvm_value}")

    def visit_If(self, node: ast.If) -> None:
        """
        if文:
          if test:
              ...
          else:
              ...
        静的に i32(0以外) をtrueとみなすなど
        """
        cond_typed: TypedValue = self.expr_visitor.visit(node.test)
        cond_val = cond_typed.llvm_value

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

    def visit_Expr(self, node: ast.Expr) -> Any:
        """
        式文
        戻り値は破棄してよいので、一応計算はするが特に変数には入れない
        """
        _ = self.expr_visitor.visit(node.value)
        # discard
        return None

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        raise NotImplementedError("Class definition not supported")

    def visit_Assign(self, node: ast.Assign) -> None:
        if len(node.targets) != 1:
            raise NotImplementedError("Multi-target assignment not supported")
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            raise NotImplementedError("Only assignment to a variable (Name) is supported")

        # 右辺を評価
        rhs_typed = self.expr_visitor.visit(node.value)  # -> TypedValue
        rhs_type = rhs_typed.type_

        var_name = f"%{target.id}"

        # 既にシンボルテーブルに型情報があるかどうか
        existing_type = self.get_symbol_type(target.id)
        if existing_type is None:
            # 初回代入 -> シンボルテーブルに登録
            self.set_symbol_type(target.id, rhs_type)
            existing_type = rhs_type
        else:
            # 型が合わない場合はエラー
            if existing_type != rhs_type:
                raise TypeError(f"Type mismatch: variable '{target.id}' is {existing_type}, but RHS is {rhs_type}")

        # ここで "代入" 相当の LLVM IR を生成
        if existing_type == "i32":
            # i32 の場合 → add i32 0, ...
            self.builder.emit(f"  {var_name} = add i32 0, {rhs_typed.llvm_value} ; assignment to {target.id}")
        elif existing_type == "ptr":
            # ptr の場合 → bitcast (もしくは単純に =)
            # LLVM IRでは「%var = bitcast ptr %rhs to ptr」が実質的に無意味
            # SSA 上「新レジスタを作る」ためにダミーのno-opを発行する:
            self.builder.emit(f"  {var_name} = bitcast ptr {rhs_typed.llvm_value} to ptr ; assignment to {target.id}")
        else:
            raise NotImplementedError(f"Unsupported type for assignment: {existing_type}")

    def visit_Delete(self, node: ast.Delete) -> None:
        raise NotImplementedError("Delete statement not supported")

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
