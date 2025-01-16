from __future__ import annotations

import ast
from typing import Any

from ..ir import IRBuilder
from .base import BaseVisitor

__all__ = ["ExprVisitor"]


class ExprVisitor(BaseVisitor, ast.NodeVisitor):
    """
    式(expr)ノードの訪問を担当するクラス
    静的型付き言語として扱う想定のため
    - int は i32 (or i64)
    - str は string用の構造体ポインタ (IR上ではptr %struct.String*など)
    などにマッピングする
    """

    def __init__(self, builder: IRBuilder):
        self.builder = builder
        # BaseVisitorには__init__が無いので super() は省略

    def get_temp_name(self) -> str:
        return self.builder.get_temp_name()

    # ---------------------------
    # visit_Constant
    # ---------------------------
    def visit_Constant(self, node: ast.Constant) -> str:
        """
        静的型に基づき、int/str等をネイティブ表現の即値や
        create_string() 呼び出しなどに変換する
        """
        val = node.value

        if isinstance(val, int):
            # i32 (または i64) 即値を作る
            tmp = self.get_temp_name()
            # IR上で i32 %tmp = <constant> 的に表現したいので
            # ここではシンプルに "add i32 0, val" を生成
            self.builder.emit(f"  {tmp} = add i32 0, {val}")
            return tmp

        elif isinstance(val, str):
            # 文字列リテラルをグローバルに置き、 create_string(ptr) を呼ぶ想定
            gname = self.builder.add_global_string(val)
            tmp = self.get_temp_name()
            self.builder.emit(f"  {tmp} = call ptr @create_string(ptr {gname})")
            return tmp

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
        例:
            - print(...) は特別扱いし、intなら print_i32(...)、
              stringなら print_string(...) を呼ぶなど。
            - ユーザ関数はシグネチャが i32->i32 かなどを型注釈から推論し、呼び出す。
        """
        if not isinstance(node.func, ast.Name):
            raise NotImplementedError("Only simple function calls supported (static)")

        func_name = node.func.id

        # ここでは例として:
        #   print(x) -> if x is i32 => call void @print_i32(i32 x)
        #               if x is ptr => call void @print_string(ptr x)
        #   それ以外 => ユーザ定義関数として対応
        if func_name == "print":
            if len(node.args) != 1:
                raise NotImplementedError("print(...) with multiple args not supported")

            arg_val = self.visit(node.args[0])  # i32 か ptr か
            # ここでは "print_i32" と "print_string" を、型判別なしでとりあえず呼べる前提
            # TODO: "どの型か" をASTか型推論で判別するようにする
            # 例: i32として扱うなら:
            tmp = self.get_temp_name()
            # IRに: call void @print_i32(i32 <arg_val>)
            self.builder.emit(f"  call void @print_i32(i32 {arg_val})")
            # とりあえず i32用のprintだけ呼ぶ
            # 戻り値はvoidだが、一応 tmp = add i32 0, 0 みたいにでっち上げる
            self.builder.emit(f"  {tmp} = add i32 0, 0")
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
