from __future__ import annotations

import ast
from typing import Any

from ..ir import IRBuilder
from .base import BaseVisitor

__all__ = ["ExprVisitor"]


class ExprVisitor(BaseVisitor, ast.NodeVisitor):
    """
    式(expr)ノードの訪問を担当するクラス
    静的型付き言語として扱う想定
    - int は i32
    - bool は i1
    - str は string用の構造体ポインタ (IR上ではptr %struct.String*など)
    などにマッピングする
    """

    def __init__(self, builder: IRBuilder):
        self.builder = builder

    def get_temp_name(self) -> str:
        return self.builder.get_temp_name()

    # ---------------------------
    # visit_Constant
    # ---------------------------
    def visit_Constant(self, node: ast.Constant) -> str:
        """
        静的型に基づき、int/str等をネイティブ表現の即値に変換する
        """
        val = node.value

        # 文字列型の生成
        if isinstance(val, str):
            gname = self.builder.add_global_string(node.value)
            tmp = self.get_temp_name()
            self.builder.emit(f"  {tmp} = call ptr @create_string(ptr {gname})")
            return tmp

        # 整数型の生成
        if isinstance(val, int):
            return str(val)  # LLVM IRでは整数リテラルはそのまま使用可能

        elif val is None:
            # None相当は静的型上廃止してもよいが null ポインタを返す
            return "null"

        else:
            raise NotImplementedError(f"Unsupported constant type: {type(val)}")

    # ---------------------------
    # visit_BinOp
    # ---------------------------
    def visit_BinOp(self, node: ast.BinOp) -> str:
        """
        2項演算
        int同士の場合は i32 同士の演算を想定
        str同士の + などは未対応としてエラーにするなど
        """
        left_val = self.visit(node.left)
        right_val = self.visit(node.right)

        # ここでは int (i32) 同士の演算を前提とする
        result_name = self.get_temp_name()

        if isinstance(node.op, ast.Add):
            self.builder.emit(f"  {result_name} = add i32 {left_val}, {right_val}")
        elif isinstance(node.op, ast.Sub):
            self.builder.emit(f"  {result_name} = sub i32 {left_val}, {right_val}")
        elif isinstance(node.op, ast.Mult):
            self.builder.emit(f"  {result_name} = mul i32 {left_val}, {right_val}")
        elif isinstance(node.op, ast.Div):
            # i32 の算術除算 (sdiv)
            self.builder.emit(f"  {result_name} = sdiv i32 {left_val}, {right_val}")
        elif isinstance(node.op, ast.Mod):
            self.builder.emit(f"  {result_name} = srem i32 {left_val}, {right_val}")
        else:
            raise NotImplementedError(f"BinOp {type(node.op)} not supported (static)")

        return result_name

    # ---------------------------
    # visit_Call
    # ---------------------------
    def visit_Call(self, node: ast.Call) -> str:
        """
        関数呼び出し。
        """
        if not isinstance(node.func, ast.Name):
            raise NotImplementedError("Only simple function calls supported (static)")

        func_name = node.func.id

        if func_name == "print":
            # 引数は1個のみ、かつ文字列型を想定
            if len(node.args) != 1:
                raise NotImplementedError("print() with multiple or zero args not supported")

            arg_val = self.visit(node.args[0])  # 文字列型(ptr)を期待
            # ここで arg_val は %struct.String* (IR上ではptr) のはず
            # 'print' 関数を呼び出す (declare void @print(ptr))
            self.builder.emit(f"  call void @print(ptr {arg_val})")

            # returnはNone(式としてはvoid)
            tmp = self.get_temp_name()
            self.builder.emit(f"  {tmp} = add i32 0, 0 ; discard return")
            return tmp

        elif func_name == "str":
            # `str()` への呼び出し
            if len(node.args) != 1:
                raise NotImplementedError("str() with multiple/zero args not supported")

            arg_val = self.visit(node.args[0])  # i32 or ptr
            # ここで "型情報" をどう取得するかは、本来型推論が必要。
            # 簡易的に、i32なら int2str、ptrなら str2str という判定を行う例:
            # (i32かptrかどうか？ => 簡易的には変数名の先頭とか、あるいはvisitor内で管理している type map による)
            # ここでは仮に "もし変数名が '%' で始まればi32" みたいな雑判定をしてみる(本当は厳密化が必要)

            # ダミーの判定例:
            if arg_val.startswith("%"):
                # i32 だと仮定
                tmp = self.builder.get_temp_name()
                self.builder.emit(f"  {tmp} = call ptr @int2str(i32 {arg_val})")
                return tmp
            else:
                # string(ptr)
                tmp = self.builder.get_temp_name()
                self.builder.emit(f"  {tmp} = call ptr @str2str(ptr {arg_val})")
                return tmp

        else:
            # ユーザー定義関数呼び出し
            # ここで引数をgatherしてIR化
            arg_vals = []
            for a in node.args:
                arg_vals.append(self.visit(a))  # i32 or ptr

            # シンプルに全部 i32 と仮定して call i32 @func_name(i32, i32, ...)
            # あるいは全部 ptr と仮定するなど。実際には型注釈が要る
            joined_args = ", ".join(f"i32 {v}" for v in arg_vals)
            ret_var = self.get_temp_name()
            # 戻り値を i32 として扱う想定
            self.builder.emit(f"  {ret_var} = call i32 @{func_name}({joined_args})")
            return ret_var

    # ---------------------------
    # visit_Compare
    # ---------------------------
    def visit_Compare(self, node: ast.Compare) -> str:
        """
        比較演算: i32同士の比較を想定して icmp.
        結果は i1 だが、boolをi32(0 or 1)に変換して返すなど。
        """
        if len(node.ops) != 1 or len(node.comparators) != 1:
            raise NotImplementedError("Only single compare op is supported (static)")

        left_val = self.visit(node.left)
        right_val = self.visit(node.comparators[0])
        op = node.ops[0]

        tmp_bool = self.get_temp_name()  # i1
        if isinstance(op, ast.LtE):
            self.builder.emit(f"  {tmp_bool} = icmp sle i32 {left_val}, {right_val}")
        elif isinstance(op, ast.Lt):
            self.builder.emit(f"  {tmp_bool} = icmp slt i32 {left_val}, {right_val}")
        elif isinstance(op, ast.GtE):
            self.builder.emit(f"  {tmp_bool} = icmp sge i32 {left_val}, {right_val}")
        elif isinstance(op, ast.Gt):
            self.builder.emit(f"  {tmp_bool} = icmp sgt i32 {left_val}, {right_val}")
        elif isinstance(op, ast.Eq):
            self.builder.emit(f"  {tmp_bool} = icmp eq i32 {left_val}, {right_val}")
        elif isinstance(op, ast.NotEq):
            self.builder.emit(f"  {tmp_bool} = icmp ne i32 {left_val}, {right_val}")
        else:
            raise NotImplementedError(f"Comparison op {type(op)} not supported")

        # i1 -> i32 へ拡張 (true=1, false=0)
        ret = self.get_temp_name()
        self.builder.emit(f"  {ret} = zext i1 {tmp_bool} to i32")
        return ret

    # ---------------------------
    # visit_Name
    # ---------------------------
    def visit_Name(self, node: ast.Name) -> str:
        """
        変数参照。
        今回は「静的型言語化」するにあたり、None/True/Falseなどを扱わない or
        それぞれ i32 0 / i32 1 などとして扱う例。
        """
        if node.id == "True":
            tmp = self.get_temp_name()
            self.builder.emit(f"  {tmp} = add i32 0, 1")
            return tmp
        elif node.id == "False":
            tmp = self.get_temp_name()
            self.builder.emit(f"  {tmp} = add i32 0, 0")
            return tmp
        elif node.id == "None":
            return "null"
        else:
            # 変数: e.g. i32 %foo
            return f"%{node.id}"

    # ------------------------------------------------
    # あと ast.NodeVisitorで要求されるgeneric_visit等
    # ------------------------------------------------
    def generic_visit(self, node: ast.AST) -> Any:
        return super().generic_visit(node)
