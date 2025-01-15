from __future__ import annotations

import ast
from typing import TYPE_CHECKING, Any

from ..ir import IRBuilder

if TYPE_CHECKING:
    from .mod import ModVisitor  # noqa

__all__ = ["ExprVisitor"]


class ExprVisitor(ast.NodeVisitor):
    def __init__(self, builder: IRBuilder):
        self.builder = builder
        self.temp_counter = builder.temp_counter

    def get_temp_name(self) -> str:
        name = self.builder.get_temp_name()
        return name

    def visit_Constant(self, node: ast.Constant) -> str:
        if isinstance(node.value, str):
            # 文字列定数の処理
            global_name = self.builder.add_global_string(node.value)
            temp_name = self.get_temp_name()
            # 文字列ポインタを取得
            self.builder.emit(
                f"  {temp_name} = call ptr @PyString_FromString(ptr {global_name})"
            )
            return temp_name
        elif isinstance(node.value, int):
            # 整数定数の処理
            temp_name = self.get_temp_name()
            self.builder.emit(
                f"  {temp_name} = call ptr @PyInt_FromLong(i64 {node.value})"
            )
            return temp_name
        elif node.value is None:
            # Noneの処理
            return "@Py_None"
        else:
            raise NotImplementedError(
                f"Constant type not supported: {type(node.value)}"
            )

    def visit_BinOp(self, node: ast.BinOp) -> str:
        # 左辺と右辺の値を取得
        left_ptr = self.visit(node.left)
        right_ptr = self.visit(node.right)

        # PyIntObjectのvalueフィールドへのポインタを取得
        left_val_ptr = self.get_temp_name()
        right_val_ptr = self.get_temp_name()
        self.builder.emit(
            f"  {left_val_ptr} = getelementptr %struct.PyIntObject, ptr {left_ptr}, i32 0, i32 1"
        )
        self.builder.emit(
            f"  {right_val_ptr} = getelementptr %struct.PyIntObject, ptr {right_ptr}, i32 0, i32 1"
        )

        # valueフィールドの値をロード
        left_val = self.get_temp_name()
        right_val = self.get_temp_name()
        self.builder.emit(f"  {left_val} = load i64, ptr {left_val_ptr}")
        self.builder.emit(f"  {right_val} = load i64, ptr {right_val_ptr}")

        # 計算結果を格納する一時変数
        result_val = self.get_temp_name()

        if isinstance(node.op, ast.Add):
            # 加算
            self.builder.emit(f"  {result_val} = add i64 {left_val}, {right_val}")
        elif isinstance(node.op, ast.Sub):
            # 減算
            self.builder.emit(f"  {result_val} = sub i64 {left_val}, {right_val}")
        elif isinstance(node.op, ast.Mult):
            # 乗算
            self.builder.emit(f"  {result_val} = mul i64 {left_val}, {right_val}")
        elif isinstance(node.op, ast.Div):
            # 除算
            self.builder.emit(f"  {result_val} = sdiv i64 {left_val}, {right_val}")
        elif isinstance(node.op, ast.Mod):
            # 剰余
            self.builder.emit(f"  {result_val} = srem i64 {left_val}, {right_val}")
        elif isinstance(node.op, ast.Pow):
            # 冪乗(難しいので後回し)
            raise NotImplementedError("Power operator not supported")
        elif isinstance(node.op, ast.LShift):
            # 左シフト
            self.builder.emit(f"  {result_val} = shl i64 {left_val}, {right_val}")
        elif isinstance(node.op, ast.RShift):
            # 右シフト
            self.builder.emit(f"  {result_val} = ashr i64 {left_val}, {right_val}")
        elif isinstance(node.op, ast.BitOr):
            # ビット論理和
            self.builder.emit(f"  {result_val} = or i64 {left_val}, {right_val}")
        elif isinstance(node.op, ast.BitXor):
            # ビット排他的論理和
            self.builder.emit(f"  {result_val} = xor i64 {left_val}, {right_val}")
        elif isinstance(node.op, ast.BitAnd):
            # ビット論理積
            self.builder.emit(f"  {result_val} = and i64 {left_val}, {right_val}")
        elif isinstance(node.op, ast.FloorDiv):
            # 切り捨て除算
            self.builder.emit(f"  {result_val} = sdiv i64 {left_val}, {right_val}")
        else:
            raise NotImplementedError(
                f"Binary operation not supported: {type(node.op)}"
            )

        # 最後に一度だけPyIntObjectを生成
        result_ptr = self.get_temp_name()
        self.builder.emit(f"  {result_ptr} = call ptr @PyInt_FromLong(i64 {result_val})")

        return result_ptr

    def visit_Call(self, node: ast.Call) -> str:
        if isinstance(node.func, ast.Name):
            if node.func.id == "print":
                if len(node.args) != 1:
                    raise NotImplementedError("print with multiple arguments not supported")
                arg_ptr = self.visit(node.args[0])
                temp_name = self.get_temp_name()
                self.builder.emit(f"  {temp_name} = call i32 @print_object(ptr {arg_ptr})")
                return temp_name
            elif node.func.id == "str":
                # str関数の呼び出し
                if len(node.args) != 1:
                    raise NotImplementedError("str with multiple arguments not supported")
                arg_ptr = self.visit(node.args[0])
                temp_name = self.get_temp_name()
                self.builder.emit(f"  {temp_name} = call ptr @str(ptr {arg_ptr})")
                return temp_name
            else:
                # ユーザー定義関数の呼び出し
                func_name = node.func.id
                args = []
                for arg in node.args:
                    arg_ptr = self.visit(arg)
                    args.append(f"ptr {arg_ptr}")

                result = self.get_temp_name()
                self.builder.emit(
                    f"  {result} = call ptr @{func_name}({', '.join(args)})"
                )
                return result
        else:
            raise NotImplementedError("Only named functions are supported")

    def generic_visit(self, node: ast.AST) -> Any:
        raise NotImplementedError(f"Unsupported expression: {type(node)}")

    def visit_Compare(self, node: ast.Compare) -> str:
        """比較演算の処理"""
        # 左辺の値を取得
        left_ptr = self.visit(node.left)

        # 現時点では単一の比較演算子のみをサポート
        if len(node.ops) != 1 or len(node.comparators) != 1:
            raise NotImplementedError("Multiple comparisons not supported")

        # 右辺の値を取得
        right_ptr = self.visit(node.comparators[0])

        # PyIntObjectのvalueフィールドへのポインタを取得
        left_val_ptr = self.get_temp_name()
        right_val_ptr = self.get_temp_name()
        self.builder.emit(
            f"  {left_val_ptr} = getelementptr %struct.PyIntObject, ptr {left_ptr}, i32 0, i32 1"
        )
        self.builder.emit(
            f"  {right_val_ptr} = getelementptr %struct.PyIntObject, ptr {right_ptr}, i32 0, i32 1"
        )

        # valueフィールドの値をロード
        left_val = self.get_temp_name()
        right_val = self.get_temp_name()
        self.builder.emit(f"  {left_val} = load i64, ptr {left_val_ptr}")
        self.builder.emit(f"  {right_val} = load i64, ptr {right_val_ptr}")

        # 比較結果を格納する一時変数（i1型）
        result = self.get_temp_name()

        # 比較演算子に応じた処理
        op = node.ops[0]
        if isinstance(op, ast.LtE):
            self.builder.emit(f"  {result} = icmp sle i64 {left_val}, {right_val}")
        elif isinstance(op, ast.Lt):
            self.builder.emit(f"  {result} = icmp slt i64 {left_val}, {right_val}")
        elif isinstance(op, ast.GtE):
            self.builder.emit(f"  {result} = icmp sge i64 {left_val}, {right_val}")
        elif isinstance(op, ast.Gt):
            self.builder.emit(f"  {result} = icmp sgt i64 {left_val}, {right_val}")
        elif isinstance(op, ast.Eq):
            self.builder.emit(f"  {result} = icmp eq i64 {left_val}, {right_val}")
        elif isinstance(op, ast.NotEq):
            self.builder.emit(f"  {result} = icmp ne i64 {left_val}, {right_val}")
        else:
            raise NotImplementedError(f"Comparison operator not supported: {type(op)}")

        # 条件分岐用のラベルを生成
        true_label = f"cmp.true.{self.builder.get_label_counter()}"
        false_label = f"cmp.false.{self.builder.get_label_counter()}"
        end_label = f"cmp.end.{self.builder.get_label_counter()}"

        # 条件分岐を生成
        self.builder.emit(f"  br i1 {result}, label %{true_label}, label %{false_label}")

        # true の場合
        self.builder.emit(f"{true_label}:")
        true_result = self.get_temp_name()
        self.builder.emit(f"  {true_result} = call ptr @PyInt_FromLong(i64 1)")
        self.builder.emit(f"  br label %{end_label}")

        # false の場合
        self.builder.emit(f"{false_label}:")
        false_result = self.get_temp_name()
        self.builder.emit(f"  {false_result} = call ptr @PyInt_FromLong(i64 0)")
        self.builder.emit(f"  br label %{end_label}")

        # 終了ラベル
        self.builder.emit(f"{end_label}:")
        result_ptr = self.get_temp_name()
        self.builder.emit(f"  {result_ptr} = phi ptr [ {true_result}, %{true_label} ], [ {false_result}, %{false_label} ]")

        return result_ptr

    def visit_Name(self, node: ast.Name) -> str:
        """変数名の処理"""
        # 現時点では変数のスコープは考慮せず、単純に変数名を返す
        if node.id == "None":
            return "@Py_None"
        elif node.id == "True":
            return "@Py_True"
        elif node.id == "False":
            return "@Py_False"
        else:
            # ローカル変数として扱う
            return f"%{node.id}"
