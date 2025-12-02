from __future__ import annotations

import ast

from lython.mlir.dialects import _lython_ops_gen as py_ops

from ..mlir import ir
from ._base import BaseVisitor, ClassInfo

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

    def __init__(
        self,
        ctx: ir.Context,
        *,
        subvisitors: dict[str, BaseVisitor],
    ) -> None:
        super().__init__(ctx, subvisitors=subvisitors)

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

    def visit_Name(self, node: ast.Name) -> ir.Value:
        if isinstance(node.ctx, ast.Store):
            raise NotImplementedError("Store context handled elsewhere")
        try:
            return self.lookup_symbol(node.id)
        except NameError:
            pass
        try:
            func_info = self.lookup_function(node.id)
        except NameError as exc:
            raise NameError(f"Variable reference '{node.id}' not implemented") from exc
        with self._loc(node), self.insertion_point():
            symbol = ir.FlatSymbolRefAttr.get(func_info.symbol, self.ctx)
            return py_ops.FuncObjectOp(func_info.func_type, symbol).result

    def visit_BinOp(self, node: ast.BinOp) -> None:
        """
        二項演算を処理する

        ```asdl
        BinOp(expr left, operator op, expr right)
        ```
        """
        lhs = self.require_value(node.left, self.visit(node.left))
        rhs = self.require_value(node.right, self.visit(node.right))
        with self._loc(node), self.insertion_point():
            if isinstance(node.op, ast.Add):
                return py_ops.NumAddOp(lhs, rhs).result
            if isinstance(node.op, ast.Sub):
                return py_ops.NumSubOp(lhs, rhs).result
        raise NotImplementedError("Unsupported binary operation")

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

    def visit_Constant(self, node: ast.Constant) -> ir.Value:
        with self._loc(node), self.insertion_point():
            if node.value is None:
                return py_ops.NoneOp(self.get_py_type("!py.none")).result
            if isinstance(node.value, bool):
                result_type = self.get_py_type("!py.bool")
                return py_ops.IntConstantOp(result_type, str(int(node.value))).result
            if isinstance(node.value, int):
                result_type = self.get_py_type("!py.int")
                # Convert to string for arbitrary precision support
                return py_ops.IntConstantOp(result_type, str(node.value)).result
            if isinstance(node.value, float):
                result_type = self.get_py_type("!py.float")
                return py_ops.FloatConstantOp(result_type, node.value).result
            if isinstance(node.value, str):
                result_type = self.get_py_type("!py.str")
                attr = ir.StringAttr.get(node.value, self.ctx)
                return py_ops.StrConstantOp(result_type, attr).result
        raise NotImplementedError(f"Unsupported constant {node.value!r}")

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

    def visit_Compare(self, node: ast.Compare) -> ir.Value:
        if len(node.ops) != 1 or len(node.comparators) != 1:
            raise NotImplementedError("Only single comparison supported")
        lhs = self.require_value(node.left, self.visit(node.left))
        rhs = self.require_value(node.comparators[0], self.visit(node.comparators[0]))
        if not isinstance(node.ops[0], ast.LtE):
            raise NotImplementedError("Only <= comparison supported")
        bool_type = self.get_py_type("!py.bool")
        with self._loc(node), self.insertion_point():
            return py_ops.NumLeOp(bool_type, lhs, rhs).result

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

    def visit_Call(self, node: ast.Call) -> ir.Value:
        """
        関数呼び出しを処理する。

        ```asdl
        Call(expr func, expr* args, keyword* keywords)
        ```
        """
        if node.keywords:
            raise NotImplementedError("Keyword arguments not supported yet")

        loc = self._loc(node)

        # Check if this is a class instantiation
        if isinstance(node.func, ast.Name):
            class_info = self.lookup_class(node.func.id)
            if class_info is not None:
                return self._handle_class_instantiation(node, class_info, loc)

        # Handle method call early to avoid generating dead py.attr.get
        if isinstance(node.func, ast.Attribute):
            return self._handle_method_call(node, loc)

        # Otherwise it's a regular function call
        callee = self.require_value(node.func, self.visit(node.func))
        arg_values = [self.require_value(arg, self.visit(arg)) for arg in node.args]

        with loc, self.insertion_point():
            if isinstance(node.func, ast.Name):
                func_info = self.lookup_function(node.func.id)
                result_types = list(func_info.result_types)
                if not func_info.has_vararg:
                    if len(arg_values) != len(func_info.arg_types):
                        raise NotImplementedError(
                            f"Function '{node.func.id}' expects {len(func_info.arg_types)} "
                            f"arguments, got {len(arg_values)}"
                        )
                    posargs = self.build_tuple(arg_values, loc=loc)
                else:
                    object_args = [
                        self.ensure_object(value, loc=loc) for value in arg_values
                    ]
                    posargs = self.build_tuple(object_args, loc=loc)
            else:
                raise NotImplementedError("Only direct function calls are supported")
            empty_tuple_type = self.get_py_type("!py.tuple<>")
            kwnames = py_ops.TupleEmptyOp(empty_tuple_type).result
            kwvalues = py_ops.TupleEmptyOp(empty_tuple_type).result
            if len(result_types) != 1:
                raise NotImplementedError("Only single-result functions supported")
            call = py_ops.CallVectorOp(result_types, callee, posargs, kwnames, kwvalues)
            return call.results_[0]

    def _handle_class_instantiation(
        self, node: ast.Call, class_info: ClassInfo, loc: ir.Location
    ) -> ir.Value:
        """Handle class instantiation: ClassName(args)"""

        arg_values = [self.require_value(arg, self.visit(arg)) for arg in node.args]

        with loc, self.insertion_point():
            # Create new instance
            instance = py_ops.ClassNewOp(class_info.class_type, class_info.name).result

            # Call __init__ if it exists
            if "__init__" in class_info.methods:
                init_info = class_info.methods["__init__"]
                # __init__ args are (self, *args) - self is already the instance
                init_args = [instance] + arg_values
                posargs = self.build_tuple(init_args, loc=loc)
                empty_tuple_type = self.get_py_type("!py.tuple<>")
                kwnames = py_ops.TupleEmptyOp(empty_tuple_type).result
                kwvalues = py_ops.TupleEmptyOp(empty_tuple_type).result

                # Get __init__ function reference
                init_funcsig = self.build_funcsig(
                    [str(t) for t in init_info.arg_types],
                    [str(t) for t in init_info.result_types],
                )
                init_func_type = self.get_py_type(f"!py.func<{init_funcsig}>")
                init_func = py_ops.FuncObjectOp(
                    init_func_type, f"{class_info.name}.__init__"
                ).result

                # Call __init__ (result is None, we ignore it)
                py_ops.CallVectorOp(
                    list(init_info.result_types),
                    init_func,
                    posargs,
                    kwnames,
                    kwvalues,
                )

            return instance

    def _handle_method_call(self, node: ast.Call, loc: ir.Location) -> ir.Value:
        """Handle method call: obj.method(args)"""
        assert isinstance(node.func, ast.Attribute)

        obj = self.require_value(node.func.value, self.visit(node.func.value))
        method_name = node.func.attr
        arg_values = [self.require_value(arg, self.visit(arg)) for arg in node.args]

        # For now, we need to determine the class from the object type
        # This is a simplified implementation
        obj_type = obj.type
        obj_type_str = str(obj_type)

        # Extract class name from type like !py.class<"Counter">
        if obj_type_str.startswith('!py.class<"') and obj_type_str.endswith('">'):
            class_name = obj_type_str[len('!py.class<"') : -len('">')]  # noqa
        else:
            raise NotImplementedError(
                f"Method calls only supported on class instances, got {obj_type_str}"
            )

        class_info = self.lookup_class(class_name)
        if class_info is None:
            raise NameError(f"Unknown class '{class_name}'")

        if method_name not in class_info.methods:
            raise AttributeError(f"Class '{class_name}' has no method '{method_name}'")

        method_info = class_info.methods[method_name]

        with loc, self.insertion_point():
            # Build args tuple with self as first argument
            method_args = [obj] + arg_values
            posargs = self.build_tuple(method_args, loc=loc)
            empty_tuple_type = self.get_py_type("!py.tuple<>")
            kwnames = py_ops.TupleEmptyOp(empty_tuple_type).result
            kwvalues = py_ops.TupleEmptyOp(empty_tuple_type).result

            # Get method function reference
            method_funcsig = self.build_funcsig(
                [str(t) for t in method_info.arg_types],
                [str(t) for t in method_info.result_types],
            )
            method_func_type = self.get_py_type(f"!py.func<{method_funcsig}>")
            method_func = py_ops.FuncObjectOp(
                method_func_type, f"{class_name}.{method_name}"
            ).result

            # Call the method
            result_types = list(method_info.result_types)
            call = py_ops.CallVectorOp(
                result_types, method_func, posargs, kwnames, kwvalues
            )
            return call.results_[0]

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

    def visit_Attribute(self, node: ast.Attribute) -> ir.Value | None:
        """
        属性アクセスを処理する

        ```asdl
        Attribute(expr value, identifier attr, expr_context ctx)
        ```
        """
        obj = self.require_value(node.value, self.visit(node.value))
        result_type = self.get_attribute_type(obj.type, node.attr)

        with self._loc(node), self.insertion_point():
            return py_ops.AttrGetOp(result_type, obj, node.attr).result

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
