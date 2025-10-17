from __future__ import annotations

import ast

from ..mlir import ir
from ._base import BaseVisitor

__all__ = ["ExprVisitor"]


class ExprVisitor(BaseVisitor):
    """
    式(expr)ノードの訪問を担当するクラス
    以下のようなオブジェクト型に対応する
    - int => PyInt (PyObject派生)
    - bool => PyBool (PyObject派生)
    - str => PyUnicodeObject (PyObject派生)
    - list => PyListObject (PyObject派生)
    - dict => PyDictObject (PyObject派生)

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

    def __init__(self, ctx: ir.Context):
        super().__init__(ctx)

    def visit_BoolOp(self, node: ast.BoolOp) -> None:
        """
        ブールの論理演算を処理する

        ```asdl
        BoolOp(boolop op, expr* values)
        ```
        """
        raise NotImplementedError("Boolean operations not implemented")

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        """
        名前付き式を処理する

        ```asdl
        NamedExpr(expr target, expr value)
        ```
        """
        raise NotImplementedError("Named expression not implemented")

    def visit_BinOp(self, node: ast.BinOp) -> None:
        """
        二項演算を処理する

        ```asdl
        BinOp(expr left, operator op, expr right)
        ```
        """
        raise NotImplementedError("Binary operation not implemented")

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
        三項演算子を処理する

        ```asdl
        IfExp(expr test, expr body, expr orelse)
        ```
        """
        raise NotImplementedError("If expression not implemented")

    def visit_Dict(self, node: ast.Dict) -> None:
        """
        辞書リテラルを処理する。
        PyDictObject*を生成し、キーと値のペアを追加する。

        ```asdl
        Dict(expr* keys, expr* values)
        ```
        """
        raise NotImplementedError("Dict literal not implemented")

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

    def visit_Compare(self, node: ast.Compare) -> None:
        """比較演算の処理"""
        raise NotImplementedError("Compare expression not implemented")

    def visit_Call(self, node: ast.Call) -> None:
        """
        関数呼び出しを処理する。

        ```asdl
        Call(expr func, expr* args, keyword* keywords)
        ```
        """
        raise NotImplementedError("Function call not implemented")

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

    def visit_Constant(self, node: ast.Constant) -> None:
        """
        定数値を処理する。
        Pythonオブジェクトとして適切に生成する。

        ```asdl
        Constant(constant value, string? kind)
        ```
        """
        raise NotImplementedError("Constant value not implemented")

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

    def visit_Name(self, node: ast.Name) -> None:
        """
        変数参照
        ```asdl
        Name(identifier id, expr_context ctx)
        ```
        """
        raise NotImplementedError("Variable reference not implemented")

    def visit_List(self, node: ast.List) -> None:
        """
        リストリテラルを処理する。
        PyListObject*を生成し、要素を追加する。

        ```asdl
        List(expr* elts, expr_context ctx)
        ```
        """
        raise NotImplementedError("List literal not implemented")

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
