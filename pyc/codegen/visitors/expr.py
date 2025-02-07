from __future__ import annotations

import ast
from typing import Any

from ..ir import IRBuilder
from .base import BaseVisitor, TypedValue
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
    ```
    """

    def __init__(self, builder: IRBuilder):
        super().__init__(builder)

    def get_temp_name(self) -> str:
        """IR上の一時変数名を生成する"""
        return self.builder.get_temp_name()

    def visit_BoolOp(self, node: ast.BoolOp) -> str:
        """
        ブールの論理演算を処理する

        ```asdl
        BoolOp(boolop op, expr* values)
        ```
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

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        """
        名前付き式を処理する

        ```asdl
        NamedExpr(expr target, expr value)
        ```
        """
        raise NotImplementedError("Named expression not implemented")

    def visit_BinOp(self, node: ast.BinOp) -> TypedValue:
        """
        二項演算を処理する

        ```asdl
        BinOp(expr left, operator op, expr right)
        ```
        """
        # 左右オペランドを再帰処理
        left_typed = self.visit(node.left)    # => TypedValue
        right_typed = self.visit(node.right)  # => TypedValue

        # i32 同士の演算でない場合は未対応 (あるいは PyInt 同士の加算にする等)
        if left_typed.type_ != "i32" or right_typed.type_ != "i32":
            # 将来的にはPyIntなどの演算にも対応できるが、ここではエラーに
            raise TypeError("BinOp: both operands must be i32 for now")

        op_visitor: OperatorVisitor = self.get_subvisitor("operator")
        result_typed = op_visitor.generate_op(node.op, left_typed, right_typed)
        return result_typed

    def visit_UnaryOp(self, node: ast.UnaryOp) -> None:
        """
        単項演算を処理する

        ```asdl
        UnaryOp(unaryop op, expr operand)
        ```
        """
        raise NotImplementedError("Unary operation not implemented")

    def visit_Lambda(self, node: ast.Lambda) -> None:
        """
        ラムダ式を処理する

        ```asdl
        Lambda(arguments args, expr body)
        ```
        """
        raise NotImplementedError("Lambda expression not implemented")

    def visit_IfExp(self, node: ast.IfExp) -> None:
        """
        条件式を処理する

        ```asdl
        IfExp(expr test, expr body, expr orelse)
        ```
        """
        raise NotImplementedError("If expression not implemented")

    def visit_Dict(self, node: ast.Dict) -> TypedValue:
        """
        辞書リテラル

        ```asdl
        Dict(expr* keys, expr* values)
        ```
        """
        num = max(len(node.keys), 8)
        temp_dict = self.get_temp_name()
        self.builder.emit(f"  {temp_dict} = call ptr @PyDict_New(i32 {num})")

        for k_node, v_node in zip(node.keys, node.values):
            k_typed = self.visit(k_node)
            v_typed = self.visit(v_node)
            self.builder.emit(f"  call i32 @PyDict_SetItem(ptr {temp_dict}, ptr {k_typed.llvm_value}, ptr {v_typed.llvm_value})")

        return TypedValue(temp_dict, "ptr")

    def visit_Set(self, node: ast.Set) -> None:
        """
        集合リテラル

        ```asdl
        Set(expr* elts)
        ```
        """
        raise NotImplementedError("Set literal not implemented")

    def visit_ListComp(self, node: ast.ListComp) -> None:
        """
        リスト内包表記を処理する

        ```asdl
        ListComp(expr elt, comprehension* generators)
        ```
        """
        raise NotImplementedError("List comprehension not implemented")

    def visit_SetComp(self, node: ast.SetComp) -> None:
        """
        集合内包表記を処理する

        ```asdl
        SetComp(expr elt, comprehension* generators)
        ```
        """
        raise NotImplementedError("Set comprehension not implemented")

    def visit_DictComp(self, node: ast.DictComp) -> None:
        """
        辞書内包表記を処理する

        ```asdl
        DictComp(expr key, expr value, comprehension* generators)
        ```
        """
        raise NotImplementedError("Dict comprehension not implemented")

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        """
        ジェネレータ式を処理する

        ```asdl
        GeneratorExp(expr elt, comprehension* generators)
        ```
        """
        raise NotImplementedError("Generator expression not implemented")

    def visit_Await(self, node: ast.Await) -> None:
        """
        await式を処理する

        ```asdl
        Await(expr value)
        ```
        """
        raise NotImplementedError("Await expression not implemented")

    def visit_Yield(self, node: ast.Yield) -> None:
        """
        yield式を処理する

        ```asdl
        Yield(expr? value)
        ```
        """
        raise NotImplementedError("Yield expression not implemented")

    def visit_YieldFrom(self, node: ast.YieldFrom) -> None:
        """
        yield from式を処理する

        ```asdl
        YieldFrom(expr value)
        ```
        """
        raise NotImplementedError("YieldFrom expression not implemented")

    def visit_Compare(self, node: ast.Compare) -> TypedValue:
        """
        比較演算を処理する

        ```asdl
        Compare(expr left, cmpop* ops, expr* comparators)
        ```
        """
        if len(node.ops) != 1 or len(node.comparators) != 1:
            raise NotImplementedError("Only single compare op is supported")

        left_typed = self.visit(node.left)  # => TypedValue
        right_typed = self.visit(node.comparators[0])  # => TypedValue

        # cmpop は "cmpop" ビジターに転送
        cmpop_visitor: CmpOpVisitor = self.get_subvisitor("cmpop")

        result_val = cmpop_visitor.generate_cmpop(
            node.ops[0],
            left_typed.llvm_value,
            right_typed.llvm_value
        )

        return TypedValue(result_val, "i32")

    def visit_Call(self, node: ast.Call) -> TypedValue:
        """
        関数の呼び出しを処理する

        ```asdl
        Call(expr func, expr* args, keyword* keywords)
        ```
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

    def sample_call_implementation(self, func_name: str, node: ast.Call) -> TypedValue:
        """
        従来の実装の中身を切り出した補助メソッド。
        """
        if func_name == "print":
            if len(node.args) != 1:
                raise NotImplementedError("print() with multiple or zero args not supported")
            arg_typed = self.visit(node.args[0])      # => TypedValue
            if arg_typed.type_ != "ptr":
                raise TypeError("print() argument must be ptr (e.g. String*)")

            # ここで llvm_value を使う
            self.builder.emit(f"  call void @print(ptr {arg_typed.llvm_value})")

            # 関数呼び出し後、i32 0 相当を戻り値にする例
            tmp = self.get_temp_name()
            self.builder.emit(f"  {tmp} = add i32 0, 0 ; discard return")
            return TypedValue(tmp, "i32")

        elif func_name == "str":
            if len(node.args) != 1:
                raise NotImplementedError("str() with multiple/zero args not supported")

            arg_val: TypedValue = self.visit(node.args[0])

            # int (i32) を文字列に変換
            if arg_val.type_ == "i32":
                tmp = self.get_temp_name()
                self.builder.emit(f"  {tmp} = call ptr @int2str(i32 {arg_val.llvm_value})")
                return TypedValue(tmp, "ptr")  # 文字列ポインタを返す

            # すでに文字列（ptr）なら、そのままコピーする
            elif arg_val.type_ == "ptr":
                tmp = self.get_temp_name()
                self.builder.emit(f"  {tmp} = call ptr @str2str(ptr {arg_val.llvm_value})")
                return TypedValue(tmp, "ptr")  # 文字列ポインタを返す

            else:
                raise TypeError(f"str() does not support type '{arg_val.type_}'")

        else:
            # ユーザー定義関数
            # シグネチャを取得
            sig = self.get_function_signature(func_name)  # -> (arg_types: [str], return_type: str
            if sig is None:
                raise NameError(f"Unknown function: '{func_name}' not found in signatures")

            arg_types, return_type = sig

            # 引数の個数をチェック
            if len(node.args) != len(arg_types):
                raise TypeError(f"Function '{func_name}' expects {len(arg_types)} args, got {len(node.args)}")

            # 引数を visit して TypedValue のリストを得る
            arg_typedvals = [self.visit(a) for a in node.args]

            # 各引数の型をシグネチャと照合し、IR用文字列を組み立て
            ir_arg_list = []
            for expected_t, actual_val in zip(arg_types, arg_typedvals):
                if actual_val.type_ != expected_t:
                    raise TypeError(
                        f"Call to '{func_name}': arg type mismatch. "
                        f"Expected {expected_t}, got {actual_val.type_}"
                    )
                ir_arg_list.append(f"{expected_t} {actual_val.llvm_value}")

            joined_args = ", ".join(ir_arg_list)

            # call命令を出力
            ret_var = self.get_temp_name()
            self.builder.emit(f"  {ret_var} = call {return_type} @{func_name}({joined_args})")

            # 戻り値を TypedValue で返す
            return TypedValue(ret_var, return_type)

    def visit_FormattedValue(self, node: ast.FormattedValue) -> None:
        """
        フォーマット済み値を処理する

        ```asdl
        FormattedValue(expr value, int? conversion, expr? format_spec)
        ```
        """
        raise NotImplementedError("Formatted value not implemented")

    def visit_JoinedStr(self, node: ast.JoinedStr) -> None:
        """
        f文字列を処理する

        ```asdl
        JoinedStr(expr* values)
        ```
        """
        raise NotImplementedError("Joined string not implemented")

    def visit_Constant(self, node: ast.Constant) -> TypedValue:
        """
        静的型に基づき、int/str等をネイティブ表現の即値に変換する
        ```asdl
        Constant(constant value, string? kind)
        ```
        """
        val = node.value
        if isinstance(val, str):
            # 文字列 -> ptr(String*)
            gname = self.builder.add_global_string(val)
            tmp = self.builder.get_temp_name()
            self.builder.emit(f"  {tmp} = call ptr @create_string(ptr {gname})")
            return TypedValue(tmp, "ptr")

        elif isinstance(val, int):
            # int -> PyInt_FromI32 -> ptr(PyInt*)
            # tmp = self.builder.get_temp_name()
            # self.builder.emit(f"  {tmp} = call ptr @PyInt_FromI32(i32 {val})")
            # return TypedValue(tmp, "ptr")
            return TypedValue(str(val), "i32")

        elif val is None:
            return TypedValue("null", "ptr")

        else:
            raise NotImplementedError(f"Unsupported constant type: {type(val)}")

    def visit_Attribute(self, node: ast.Attribute) -> None:
        """
        属性アクセスを処理する

        ```asdl
        Attribute(expr value, identifier attr, expr_context ctx)
        ```
        """
        raise NotImplementedError("Attribute access not implemented")

    def visit_Subscript(self, node: ast.Subscript) -> None:
        """
        添字アクセスを処理する

        ```asdl
        Subscript(expr value, expr slice, expr_context ctx)
        ```
        """
        raise NotImplementedError("Subscript access not implemented")

    def visit_Starred(self, node: ast.Starred) -> None:
        """
        a, b* = it のようなスター式を処理する

        ```asdl
        Starred(expr value, expr_context ctx)
        ```
        """
        raise NotImplementedError("Starred expression not implemented")

    def visit_Name(self, node: ast.Name) -> TypedValue:
        """
        変数参照
        ```asdl
        Name(identifier id, expr_context ctx)
        ```
        """
        # True, False, None など特別対応
        if node.id == "True":
            t = self.get_temp_name()
            self.builder.emit(f"  {t} = add i32 0, 1")
            return TypedValue(t, "i32")

        elif node.id == "False":
            t = self.get_temp_name()
            self.builder.emit(f"  {t} = add i32 0, 0")
            return TypedValue(t, "i32")

        elif node.id == "None":
            return TypedValue("null", "ptr")

        else:
            # 変数ならシンボルテーブルから型を取得
            var_type = self.get_symbol_type(node.id)
            if var_type is None:
                raise NameError(f"Use of variable '{node.id}' before assignment")

            # 変数をレジスタ名として返す
            # ただし実際には store/load が必要だが、ここではレジスタ運用(単純化)
            return TypedValue(f"%{node.id}", var_type)

    def visit_List(self, node: ast.List) -> TypedValue:
        """
        リストリテラル
        ```asdl
        List(expr* elts, expr_context ctx)
        ```
        """
        num = max(len(node.elts), 8)
        temp_list = self.get_temp_name()
        self.builder.emit(f"  {temp_list} = call ptr @PyList_New(i32 {num})")

        for elt in node.elts:
            elt_typed = self.visit(elt)  # -> TypedValue
            # リストには ptr で格納する
            self.builder.emit(f"  call i32 @PyList_Append(ptr {temp_list}, ptr {elt_typed.llvm_value})")

        return TypedValue(temp_list, "ptr")

    def visit_Tuple(self, node: ast.Tuple) -> None:
        """
        タプルリテラル
        ```asdl
        Tuple(expr* elts, expr_context ctx)
        ```
        """
        raise NotImplementedError("Tuple literal not implemented")

    def visit_Slice(self, node: ast.Slice) -> None:
        """
        スライス式
        ```asdl
        Slice(expr? lower, expr? upper, expr? step)
        ```
        """
        raise NotImplementedError("Slice expression not implemented")

    def generic_visit(self, node: ast.AST) -> Any:
        return super().generic_visit(node)
