from __future__ import annotations

import ast
from typing import Any

from ..ir import IRBuilder
from .base import BaseVisitor
from .boolop import BoolOpVisitor
from .cmpop import CmpOpVisitor
from .keyword import KeywordVisitor
from .operator import OperatorVisitor

__all__ = ["ExprVisitor"]


class ExprVisitor(BaseVisitor, ast.NodeVisitor):
    """
    式(expr)ノードの訪問を担当するクラス
    静的型付き言語として扱う想定
    - int は i32
    - bool は i1
    - str は string用の構造体ポインタ (IR上ではptr %struct.String*など)
    などにマッピングする

    ```asdl
          -- BoolOp() can use left & right?
    expr = BoolOp(boolop op, expr* values)
         | NamedExpr(expr target, expr value)
         | BinOp(expr left, operator op, expr right)
         | UnaryOp(unaryop op, expr operand)
         | Lambda(arguments args, expr body)
         | IfExp(expr test, expr body, expr orelse)
         | Dict(expr* keys, expr* values)
         | Set(expr* elts)
         | ListComp(expr elt, comprehension* generators)
         | SetComp(expr elt, comprehension* generators)
         | DictComp(expr key, expr value, comprehension* generators)
         | GeneratorExp(expr elt, comprehension* generators)
         -- the grammar constrains where yield expressions can occur
         | Await(expr value)
         | Yield(expr? value)
         | YieldFrom(expr value)
         -- need sequences for compare to distinguish between
         -- x < 4 < 3 and (x < 4) < 3
         | Compare(expr left, cmpop* ops, expr* comparators)
         | Call(expr func, expr* args, keyword* keywords)
         | FormattedValue(expr value, int conversion, expr? format_spec)
         | JoinedStr(expr* values)
         | Constant(constant value, string? kind)

         -- the following expression can appear in assignment context
         | Attribute(expr value, identifier attr, expr_context ctx)
         | Subscript(expr value, expr slice, expr_context ctx)
         | Starred(expr value, expr_context ctx)
         | Name(identifier id, expr_context ctx)
         | List(expr* elts, expr_context ctx)
         | Tuple(expr* elts, expr_context ctx)

         -- can appear only in Subscript
         | Slice(expr? lower, expr? upper, expr? step)

          -- col_offset is the byte offset in the utf8 string the parser uses
          attributes (int lineno, int col_offset, int? end_lineno, int? end_col_offset)
    """

    def __init__(self, builder: IRBuilder):
        super().__init__(builder)

    def get_temp_name(self) -> str:
        """IR上の一時変数名を生成する"""
        return self.builder.get_temp_name()

    def visit_BoolOp(self, node: ast.BoolOp) -> str:
        """
        ```asdl
        BoolOp(boolop op, expr* values)
        """
        # boolop 自体(And/Or)は "boolop" ビジターに転送
        boolop_visitor: BoolOpVisitor = self.get_subvisitor("boolop")
        # まず boolop を処理
        op_repr = boolop_visitor.visit(node.op)
        # node.values は [expr, expr, ...]
        # ここでは例として2項 (Pythonは多項And/Orだが簡略化)
        if len(node.values) != 2:
            raise NotImplementedError("Multi-value BoolOp not fully supported")

        left_val = self.visit(node.values[0])
        right_val = self.visit(node.values[1])

        # (仮) i32として 0, 1 で計算し AND or OR する
        temp = self.get_temp_name()
        if op_repr == "And":
            self.builder.emit(f"  {temp} = and i32 {left_val}, {right_val}")
        elif op_repr == "Or":
            self.builder.emit(f"  {temp} = or i32 {left_val}, {right_val}")
        else:
            raise NotImplementedError(f"Unsupported BoolOp: {op_repr}")

        return temp

    def visit_BinOp(self, node: ast.BinOp) -> str:
        """
        ```asdl
        BinOp(expr left, operator op, expr right)
        """
        left_val = self.visit(node.left)
        right_val = self.visit(node.right)

        # operator ノード (Add, Sub, etc.) を "operator" ビジターに転送
        operator_visitor: OperatorVisitor = self.get_subvisitor("operator")
        op_str = operator_visitor.visit(node.op)
        # ここでは example: "Add", "Sub", ...

        result_name = self.get_temp_name()

        if op_str == "Add":
            self.builder.emit(f"  {result_name} = add i32 {left_val}, {right_val}")
        elif op_str == "Sub":
            self.builder.emit(f"  {result_name} = sub i32 {left_val}, {right_val}")
        elif op_str == "Mult":
            self.builder.emit(f"  {result_name} = mul i32 {left_val}, {right_val}")
        elif op_str == "Div":
            self.builder.emit(f"  {result_name} = sdiv i32 {left_val}, {right_val}")
        elif op_str == "Mod":
            self.builder.emit(f"  {result_name} = srem i32 {left_val}, {right_val}")
        else:
            raise NotImplementedError(f"BinOp {op_str} not supported")

        return result_name

    def visit_Compare(self, node: ast.Compare) -> str:
        """
        ```asdl
        Compare(expr left, cmpop* ops, expr* comparators)
        """
        # cmpopは "cmpop" ビジターに転送
        cmpop_visitor: CmpOpVisitor = self.get_subvisitor("cmpop")

        if len(node.ops) != 1 or len(node.comparators) != 1:
            raise NotImplementedError("Multi-compare chain not supported")

        left_val = self.visit(node.left)
        op_str = cmpop_visitor.visit(node.ops[0])
        right_val = self.visit(node.comparators[0])

        tmp_bool = self.get_temp_name()  # i1

        # 例: eq, ne, lt, etc.
        if op_str == "LtE":
            self.builder.emit(f"  {tmp_bool} = icmp sle i32 {left_val}, {right_val}")
        elif op_str == "Lt":
            self.builder.emit(f"  {tmp_bool} = icmp slt i32 {left_val}, {right_val}")
        elif op_str == "GtE":
            self.builder.emit(f"  {tmp_bool} = icmp sge i32 {left_val}, {right_val}")
        elif op_str == "Gt":
            self.builder.emit(f"  {tmp_bool} = icmp sgt i32 {left_val}, {right_val}")
        elif op_str == "Eq":
            self.builder.emit(f"  {tmp_bool} = icmp eq i32 {left_val}, {right_val}")
        elif op_str == "NotEq":
            self.builder.emit(f"  {tmp_bool} = icmp ne i32 {left_val}, {right_val}")
        else:
            raise NotImplementedError(f"Comparison op {op_str} not supported")

        # i1 -> i32 に zext
        ret = self.get_temp_name()
        self.builder.emit(f"  {ret} = zext i1 {tmp_bool} to i32")
        return ret

    def visit_Call(self, node: ast.Call) -> str:
        """
        ```asdl
        Call(expr func, expr* args, keyword* keywords)
        """
        # 引数: node.args
        # キーワード: node.keywords -> "keyword"ビジターに転送する
        # 仮で func が単なる Name の場合の処理を継承
        if not isinstance(node.func, ast.Name):
            raise NotImplementedError("Only simple function calls supported (static)")

        func_name = node.func.id

        # キーワード引数 (例: print(x, sep=' ') など) は "keyword"ビジターへ
        if node.keywords:
            keyword_visitor: KeywordVisitor = self.get_subvisitor("keyword")
            for kw in node.keywords:
                # 仮
                keyword_visitor.visit(kw)

        # 従来の "print" や "str" などの実装を流用
        return self.sample_call_implementation(func_name, node)

    def sample_call_implementation(self, func_name: str, node: ast.Call) -> str:
        """
        従来の実装の中身を切り出した補助メソッド。
        """
        if func_name == "print":
            if len(node.args) != 1:
                raise NotImplementedError("print() with multiple or zero args not supported")
            arg_val = self.visit(node.args[0])
            self.builder.emit(f"  call void @print(ptr {arg_val})")
            tmp = self.get_temp_name()
            self.builder.emit(f"  {tmp} = add i32 0, 0 ; discard return")
            return tmp

        elif func_name == "str":
            if len(node.args) != 1:
                raise NotImplementedError("str() with multiple/zero args not supported")
            arg_val = self.visit(node.args[0])
            # TODO: 型情報で判断するべき
            if arg_val.startswith("%"):
                tmp = self.get_temp_name()
                self.builder.emit(f"  {tmp} = call ptr @int2str(i32 {arg_val})")
                return tmp
            else:
                tmp = self.get_temp_name()
                self.builder.emit(f"  {tmp} = call ptr @str2str(ptr {arg_val})")
                return tmp

        else:
            # ユーザー定義関数
            arg_vals = []
            for a in node.args:
                arg_vals.append(self.visit(a))
            joined_args = ", ".join(f"i32 {v}" for v in arg_vals)
            ret_var = self.get_temp_name()
            self.builder.emit(f"  {ret_var} = call i32 @{func_name}({joined_args})")
            return ret_var

    def visit_Constant(self, node: ast.Constant) -> str:
        """
        静的型に基づき、int/str等をネイティブ表現の即値に変換する
        ```asdl
        Constant(constant value, string? kind)
        """
        val = node.value
        if isinstance(val, str):
            gname = self.builder.add_global_string(val)
            tmp = self.get_temp_name()
            self.builder.emit(f"  {tmp} = call ptr @create_string(ptr {gname})")
            return tmp
        elif isinstance(val, int):
            return str(val)
        elif val is None:
            return "null"
        else:
            raise NotImplementedError(f"Unsupported constant type: {type(val)}")

    def visit_Name(self, node: ast.Name) -> str:
        """
        変数参照。
        ```asdl
        Name(identifier id, expr_context ctx)
        """
        # True, False, None など簡略対応
        if node.id == "True":
            t = self.get_temp_name()
            self.builder.emit(f"  {t} = add i32 0, 1")
            return t
        elif node.id == "False":
            t = self.get_temp_name()
            self.builder.emit(f"  {t} = add i32 0, 0")
            return t
        elif node.id == "None":
            return "null"
        else:
            # 変数: i32 %foo など
            return f"%{node.id}"

    def generic_visit(self, node: ast.AST) -> Any:
        return super().generic_visit(node)
