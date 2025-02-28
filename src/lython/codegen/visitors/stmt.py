from __future__ import annotations

import ast
from typing import Any

from ..ir import IRBuilder
from .base import BaseVisitor, TypedValue
from .expr import ExprVisitor

__all__ = ["StmtVisitor"]


class StmtVisitor(BaseVisitor):
    """
    文(stmt)ノードを訪問しIRを生成するクラス
    オブジェクトシステムと連携して、Python オブジェクトの操作を行う

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

    def __init__(self, builder: IRBuilder):
        super().__init__(builder)
        self.expr_visitor = ExprVisitor(builder)
        self.current_return_type = "i32"
        self.current_return_python_type = "int"
        self.current_function_is_object_return = False

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
        arg_names = []
        arg_types = []
        arg_python_types = []

        for arg in node.args.args:
            llvm_type = "ptr"  # デフォルトはオブジェクト型
            python_type = "object"
            is_object = True

            if arg.annotation:
                if isinstance(arg.annotation, ast.Name):
                    python_type = arg.annotation.id
                    if python_type == "int":
                        llvm_type = "i32"
                        is_object = False
                    elif python_type == "str":
                        llvm_type = "ptr"
                        is_object = True
                    elif python_type == "bool":
                        llvm_type = "i1"
                        is_object = False
                    elif python_type in self.builder.known_python_types:
                        llvm_type = "ptr"
                        is_object = True
                    else:
                        llvm_type = "ptr"
                        is_object = True
                else:
                    raise NotImplementedError(f"Complex type annotations are not supported: {arg.annotation}")

            # 引数はポインタではなく値として登録
            self.set_symbol_type(arg.arg, llvm_type, python_type, is_object)
            arg_types.append(llvm_type)
            arg_python_types.append(python_type)
            arg_names.append(arg.arg)

        # 戻り値型を解析
        return_llvm_type = "ptr"  # デフォルトはオブジェクト型
        return_python_type = "object"
        is_object_return = True

        if node.returns is not None and isinstance(node.returns, ast.Name):
            return_python_type = node.returns.id
            # ビルダーの型レジストリを使用
            return_llvm_type = self.builder.get_python_type_llvm(return_python_type)
            is_object_return = return_llvm_type == "ptr" and return_python_type != "None"
        elif node.returns is None:
            # 戻り値型が指定されていない場合はNoneを返す
            return_llvm_type = "ptr"
            return_python_type = "None"
            is_object_return = True

        # この関数のシグネチャを登録
        self.set_function_signature(
            func_name, arg_types, return_llvm_type, arg_python_types, return_python_type
        )

        # 現在の関数の戻り値型を更新
        self.current_return_type = return_llvm_type
        self.current_return_python_type = return_python_type
        self.current_function_is_object_return = is_object_return

        # IR出力
        self.builder.emit("")
        self.builder.emit(f"; Function definition: {func_name}")

        # LLVM IR の仮引数部
        joined_args = ", ".join(f"{t} %{arg_name}" for t, arg_name in zip(arg_types, arg_names))
        self.builder.emit(f"define {return_llvm_type} @{func_name}({joined_args}) #0 {{")
        self.builder.emit("entry:")

        # 関数ボディの処理
        has_return = False
        for s in node.body:
            if isinstance(s, ast.Return):
                has_return = True
            self.visit(s)

        # 明示的なreturnがない場合、デフォルト値を返す
        if not has_return:
            if return_llvm_type == "i32":
                self.builder.emit("  ret i32 0")
            elif return_llvm_type == "i1":
                self.builder.emit("  ret i1 false")
            elif return_llvm_type == "ptr":
                if return_python_type == "None":
                    # None を返す
                    tmp = self.builder.get_temp_name()
                    self.builder.emit(f"  {tmp} = load ptr, ptr @Py_None")
                    self.builder.emit(f"  call void @Py_INCREF(ptr {tmp})")
                    self.builder.emit(f"  ret ptr {tmp}")
                else:
                    # 未定義の場合は NULL を返す
                    self.builder.emit("  ret ptr null")
            else:
                self.builder.emit("  ret void")

        self.builder.emit("}")

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
        if node.value is None:
            # デフォルトの戻り値
            if self.current_return_type == "i32":
                self.builder.emit("  ret i32 0")
            elif self.current_return_type == "i1":
                self.builder.emit("  ret i1 false")
            elif self.current_return_type == "ptr":
                if self.current_return_python_type == "None":
                    # None を返す
                    tmp = self.builder.get_temp_name()
                    self.builder.emit(f"  {tmp} = load ptr, ptr @Py_None")
                    self.builder.emit(f"  call void @Py_INCREF(ptr {tmp})")
                    self.builder.emit(f"  ret ptr {tmp}")
                else:
                    # 未定義の場合は NULL を返す
                    self.builder.emit("  ret ptr null")
            else:
                self.builder.emit("  ret void")
        else:
            # 式を評価して戻り値を取得
            val_typed = self.expr_visitor.visit(node.value)

            # 戻り値の型変換が必要な場合
            if val_typed.type_ != self.current_return_type:
                if self.current_function_is_object_return and not val_typed.is_object:
                    # プリミティブ型をオブジェクト型に変換（ボクシング）
                    val_typed = self.get_boxed_value(val_typed)
                elif not self.current_function_is_object_return and val_typed.is_object:
                    # オブジェクト型をプリミティブ型に変換（アンボクシング）
                    val_typed = self.get_unboxed_value(val_typed, self.current_return_type)
                else:
                    # 型変換が必要な場合は警告
                    self.builder.emit(f"  ; Warning: Return type mismatch, expected {self.current_return_type}, got {val_typed.type_}")

            # 戻り値を返す
            self.builder.emit(f"  ret {val_typed.type_} {val_typed.llvm_value}")

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
        if len(node.targets) != 1:
            raise NotImplementedError("Assignment to multiple targets is not supported")

        target = node.targets[0]
        if not isinstance(target, ast.Name):
            raise NotImplementedError("Assignment to non-name targets is not supported")

        # 右辺を評価
        rhs_typed = self.expr_visitor.visit(node.value)
        var_name = target.id

        # 既存の型情報を取得
        existing_type_info = self.get_symbol_type(var_name)

        if existing_type_info is None:
            # 初回代入の場合
            llvm_type = rhs_typed.type_
            python_type = rhs_typed.python_type
            is_object = rhs_typed.is_object

            # シンボルテーブルに登録
            self.set_symbol_type(var_name, llvm_type, python_type, is_object)

            # 代入操作
            self.builder.emit(f"  %{var_name} = alloca {llvm_type}")
            self.builder.emit(f"  store {llvm_type} {rhs_typed.llvm_value}, ptr %{var_name}")

            # オブジェクトの場合は参照カウント増加
            if is_object:
                self.builder.emit(f"  call void @Py_INCREF(ptr {rhs_typed.llvm_value})")

        else:
            # 再代入の場合
            llvm_type, python_type, is_object = existing_type_info

            # 型変換が必要な場合
            if rhs_typed.type_ != llvm_type:
                if is_object and not rhs_typed.is_object:
                    # プリミティブ型をオブジェクト型に変換（ボクシング）
                    rhs_typed = self.get_boxed_value(rhs_typed)
                elif not is_object and rhs_typed.is_object:
                    # オブジェクト型をプリミティブ型に変換（アンボクシング）
                    rhs_typed = self.get_unboxed_value(rhs_typed, llvm_type)
                else:
                    # その他の型変換
                    tmp = self.builder.get_temp_name()
                    self.builder.emit(f"  {tmp} = bitcast {rhs_typed.type_} {rhs_typed.llvm_value} to {llvm_type}")
                    rhs_typed = TypedValue(tmp, llvm_type, python_type, is_object)

            # オブジェクトの場合は古い値の参照カウント減少
            if is_object:
                old_val = self.builder.get_temp_name()
                # 修正: ptr* ではなく ptr を使用
                self.builder.emit(f"  {old_val} = load {llvm_type}, ptr %{var_name}")
                self.builder.emit(f"  call void @Py_DECREF(ptr {old_val})")

            # 新しい値を格納
            self.builder.emit(f"  store {llvm_type} {rhs_typed.llvm_value}, ptr %{var_name}")

            # オブジェクトの場合は新しい値の参照カウント増加
            if is_object:
                self.builder.emit(f"  call void @Py_INCREF(ptr {rhs_typed.llvm_value})")

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
        raise NotImplementedError("An assignment with a type annotation is not implemented")

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
        # 条件を評価
        cond_typed = self.expr_visitor.visit(node.test)

        # 条件がすでにi1型（ブール値）でない場合は変換
        if cond_typed.type_ != "i1":
            if cond_typed.type_ == "i32":
                # 整数を真偽値に変換
                tmp = self.builder.get_temp_name()
                self.builder.emit(f"  {tmp} = icmp ne i32 {cond_typed.llvm_value}, 0")
                cond_typed = TypedValue(tmp, "i1", "bool", False)
            elif cond_typed.is_object:
                # オブジェクトを真偽値に変換
                tmp = self.builder.get_temp_name()
                self.builder.emit(f"  {tmp} = call i32 @PyObject_IsTrue(ptr {cond_typed.llvm_value})")
                bool_tmp = self.builder.get_temp_name()
                self.builder.emit(f"  {bool_tmp} = icmp ne i32 {tmp}, 0")
                cond_typed = TypedValue(bool_tmp, "i1", "bool", False)

        # 分岐ラベルを生成
        then_label = f"if.then.{self.builder.get_label_counter()}"
        else_label = f"if.else.{self.builder.get_label_counter()}"
        end_label = f"if.end.{self.builder.get_label_counter()}"

        # 分岐命令
        self.builder.emit(f"  br i1 {cond_typed.llvm_value}, label %{then_label}, label %{else_label}")

        # Then節
        self.builder.emit(f"{then_label}:")
        has_then_return = False
        for stmt in node.body:
            if isinstance(stmt, ast.Return):
                has_then_return = True
            self.visit(stmt)
        if not has_then_return:
            self.builder.emit(f"  br label %{end_label}")

        # Else節
        self.builder.emit(f"{else_label}:")
        has_else_return = False
        for stmt in node.orelse:
            if isinstance(stmt, ast.Return):
                has_else_return = True
            self.visit(stmt)
        if not has_else_return:
            self.builder.emit(f"  br label %{end_label}")

        # 終了ラベル
        if not (has_then_return and has_else_return):  # 両方のパスでreturnがある場合は不要
            self.builder.emit(f"{end_label}:")

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
        # 式を評価し、戻り値を破棄
        typed_val = self.expr_visitor.visit(node.value)

        # オブジェクトの場合は参照カウント減少（評価結果が一時的に増加するため）
        if typed_val.is_object:
            self.builder.emit(f"  call void @Py_DECREF(ptr {typed_val.llvm_value})")

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
        raise NotImplementedError("Continue statement not implemeted")

    def generic_visit(self, node: ast.AST) -> None:
        return super().generic_visit(node)
