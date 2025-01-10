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
        self.temp_counter = 0

    def get_temp_name(self) -> str:
        name = f"%t{self.temp_counter}"
        self.temp_counter += 1
        return name

    def visit_Constant(self, node: ast.Constant) -> str:
        if isinstance(node.value, str):
            # 文字列定数の処理
            global_name = self.builder.add_global_string(node.value)
            # getelementptr命令を使用して文字列へのポインタを取得
            temp_name = self.get_temp_name()
            self.builder.emit(
                f"  {temp_name} = getelementptr [{len(node.value) + 1} x i8], "
                f"ptr {global_name}, i64 0, i64 0"
            )
            return temp_name
        elif isinstance(node.value, int):
            # 整数定数の処理
            temp_name = self.get_temp_name()
            self.builder.emit(f"  {temp_name} = call ptr @PyInt_FromLong(i64 {node.value})")
            return temp_name
        elif node.value is None:
            # Noneの処理
            return "null"
        else:
            raise NotImplementedError(f"Constant type not supported: {type(node.value)}")

    def visit_BinOp(self, node: ast.BinOp) -> str:
        # 左辺と右辺の値を取得
        left_ptr = self.visit(node.left)
        right_ptr = self.visit(node.right)

        # PyIntオブジェクトから整数値を取り出す
        left_val = self.get_temp_name()
        right_val = self.get_temp_name()
        self.builder.emit(f"  {left_val} = load i64, ptr {left_ptr}")
        self.builder.emit(f"  {right_val} = load i64, ptr {right_ptr}")

        # 計算結果を格納する一時変数
        temp_name = self.get_temp_name()
        # PyIntオブジェクトを格納する一時変数
        result_name = self.get_temp_name()

        if isinstance(node.op, ast.Add):
            # 加算
            self.builder.emit(f"  {temp_name} = add i64 {left_val}, {right_val}")
            self.builder.emit(f"  {result_name} = call ptr @PyInt_FromLong(i64 {temp_name})")
        elif isinstance(node.op, ast.Sub):
            # 減算
            self.builder.emit(f"  {temp_name} = sub i64 {left_val}, {right_val}")
            self.builder.emit(f"  {result_name} = call ptr @PyInt_FromLong(i64 {temp_name})")
        elif isinstance(node.op, ast.Mult):
            # 乗算
            self.builder.emit(f"  {temp_name} = mul i64 {left_val}, {right_val}")
            self.builder.emit(f"  {result_name} = call ptr @PyInt_FromLong(i64 {temp_name})")
        elif isinstance(node.op, ast.Div):
            # 除算（整数除算）
            self.builder.emit(f"  {temp_name} = sdiv i64 {left_val}, {right_val}")
            self.builder.emit(f"  {result_name} = call ptr @PyInt_FromLong(i64 {temp_name})")
        elif isinstance(node.op, ast.Mod):
            # 剰余
            self.builder.emit(f"  {temp_name} = srem i64 {left_val}, {right_val}")
            self.builder.emit(f"  {result_name} = call ptr @PyInt_FromLong(i64 {temp_name})")
        elif isinstance(node.op, ast.Pow):
            # べき乗(難しいので実装は後回し)
            raise NotImplementedError("Power operation not supported")
        elif isinstance(node.op, ast.LShift):
            # 左シフト
            self.builder.emit(f"  {temp_name} = shl i64 {left_val}, {right_val}")
            self.builder.emit(f"  {result_name} = call ptr @PyInt_FromLong(i64 {temp_name})")
        elif isinstance(node.op, ast.RShift):
            # 右シフト
            self.builder.emit(f"  {temp_name} = ashr i64 {left_val}, {right_val}")
            self.builder.emit(f"  {result_name} = call ptr @PyInt_FromLong(i64 {temp_name})")
        elif isinstance(node.op, ast.BitOr):
            # ビット論理和
            self.builder.emit(f"  {temp_name} = or i64 {left_val}, {right_val}")
            self.builder.emit(f"  {result_name} = call ptr @PyInt_FromLong(i64 {temp_name})")
        elif isinstance(node.op, ast.BitXor):
            # ビット排他的論理和
            self.builder.emit(f"  {temp_name} = xor i64 {left_val}, {right_val}")
            self.builder.emit(f"  {result_name} = call ptr @PyInt_FromLong(i64 {temp_name})")
        elif isinstance(node.op, ast.BitAnd):
            # ビット論理積
            self.builder.emit(f"  {temp_name} = and i64 {left_val}, {right_val}")
            self.builder.emit(f"  {result_name} = call ptr @PyInt_FromLong(i64 {temp_name})")
        elif isinstance(node.op, ast.FloorDiv):
            # 切り捨て除算
            self.builder.emit(f"  {temp_name} = sdiv i64 {left_val}, {right_val}")
            self.builder.emit(f"  {result_name} = call ptr @PyInt_FromLong(i64 {temp_name})")
        else:
            raise NotImplementedError(f"Binary operation not supported: {type(node.op)}")

        return result_name

    def visit_Call(self, node: ast.Call) -> str:
        if isinstance(node.func, ast.Name):
            if node.func.id == "print":
                # print関数の呼び出しを生成
                if not node.args:
                    raise ValueError("print function requires at least one argument")
                arg = self.visit(node.args[0])
                # i32 @puts(i8* %str)
                temp_name = self.get_temp_name()
                self.builder.emit(f"  {temp_name} = call i32 @print(ptr {arg})")
                return temp_name
            elif node.func.id == "int":
                # int関数の呼び出しを生成
                if not node.args:
                    raise ValueError("int function requires at least one argument")
                arg = self.visit(node.args[0])
                temp_name = self.get_temp_name()
                self.builder.emit(f"  {temp_name} = call ptr @PyInt_FromLong(i64 {arg})")
                return temp_name
            elif node.func.id == "str":
                # str関数の呼び出しを生成
                if not node.args:
                    raise ValueError("str function requires at least one argument")
                arg = self.visit(node.args[0])
                temp_name = self.get_temp_name()
                self.builder.emit(f"  {temp_name} = call ptr @PyString_FromString(ptr {arg})")
                return temp_name
            else:
                raise NotImplementedError(f"Function not supported: {node.func.id}")
        else:
            raise NotImplementedError(f"Unsupported function call: {type(node.func)}")

    def generic_visit(self, node: ast.AST) -> Any:
        raise NotImplementedError(f"Unsupported expression: {type(node)}")
