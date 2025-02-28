from __future__ import annotations

import ast
from typing import Any

from ..ir import IRBuilder
from .base import BaseVisitor, TypedValue
from .operator import OperatorVisitor

__all__ = ["ExprVisitor"]


class ExprVisitor(BaseVisitor, ast.NodeVisitor):
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

    def __init__(self, builder: IRBuilder):
        super().__init__(builder)

    def get_temp_name(self) -> str:
        """IR上の一時変数名を生成する"""
        return self.builder.get_temp_name()

    def visit_BoolOp(self, node: ast.BoolOp) -> TypedValue:
        """
        ブールの論理演算を処理する

        ```asdl
        BoolOp(boolop op, expr* values)
        ```
        """
        from .boolop import BoolOpVisitor
        boolop_visitor: BoolOpVisitor = self.get_subvisitor("boolop")
        return boolop_visitor.visit(node, self)

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

        # OperatorVisitorに処理を委譲
        op_visitor: OperatorVisitor = self.get_subvisitor("operator")
        return op_visitor.generate_op(node.op, left_typed, right_typed)

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

    def visit_Dict(self, node: ast.Dict) -> TypedValue:
        """
        辞書リテラルを処理する。
        PyDictObject*を生成し、キーと値のペアを追加する。

        ```asdl
        Dict(expr* keys, expr* values)
        ```
        """
        # 新しい辞書を作成
        temp_dict = self.get_temp_name()
        self.builder.emit(f"  {temp_dict} = call ptr @PyDict_New()")

        # 各キーと値のペアを追加
        for k_node, v_node in zip(node.keys, node.values):
            # キーと値を評価
            k_typed = self.visit(k_node)
            v_typed = self.visit(v_node)

            # オブジェクトとして扱う
            k_obj = self.ensure_object(k_typed)
            v_obj = self.ensure_object(v_typed)

            # 辞書に追加（参照カウントは自動で増加）
            self.builder.emit(f"  call i32 @PyDict_SetItem(ptr {temp_dict}, ptr {k_obj.llvm_value}, ptr {v_obj.llvm_value})")

        return TypedValue.create_object(temp_dict, "dict")

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
        """比較演算の処理"""
        if len(node.ops) != 1 or len(node.comparators) != 1:
            raise NotImplementedError("Only single compare op is supported")

        # 左右のオペランドを評価
        left_typed = self.visit(node.left)
        right_typed = self.visit(node.comparators[0])
        op = node.ops[0]

        # プリミティブ型の直接比較 (特に整数の場合)
        if not left_typed.is_object and not right_typed.is_object:
            if left_typed.type_ == "i32" and right_typed.type_ == "i32":
                result = self.builder.get_temp_name()

                # 整数の比較は直接LLVM比較命令を使用
                if isinstance(op, ast.Eq):
                    self.builder.emit(f"  {result} = icmp eq i32 {left_typed.llvm_value}, {right_typed.llvm_value}")
                elif isinstance(op, ast.NotEq):
                    self.builder.emit(f"  {result} = icmp ne i32 {left_typed.llvm_value}, {right_typed.llvm_value}")
                elif isinstance(op, ast.Lt):
                    self.builder.emit(f"  {result} = icmp slt i32 {left_typed.llvm_value}, {right_typed.llvm_value}")
                elif isinstance(op, ast.LtE):
                    self.builder.emit(f"  {result} = icmp sle i32 {left_typed.llvm_value}, {right_typed.llvm_value}")
                elif isinstance(op, ast.Gt):
                    self.builder.emit(f"  {result} = icmp sgt i32 {left_typed.llvm_value}, {right_typed.llvm_value}")
                elif isinstance(op, ast.GtE):
                    self.builder.emit(f"  {result} = icmp sge i32 {left_typed.llvm_value}, {right_typed.llvm_value}")
                else:
                    raise NotImplementedError(f"Comparison operator {type(op).__name__} not implemented for primitive types")

                return TypedValue.create_primitive(result, "i1", "bool")

        # それ以外はサブビジターに委譲
        from .cmpop import CmpOpVisitor
        cmpop_visitor: CmpOpVisitor = self.get_subvisitor("cmpop")
        return cmpop_visitor.visit(node, self)

    def visit_Call(self, node: ast.Call) -> TypedValue:
        """
        関数呼び出しを処理する。

        ```asdl
        Call(expr func, expr* args, keyword* keywords)
        ```
        """
        # 基本の関数呼び出し処理（print, strなど）
        if isinstance(node.func, ast.Name):
            func_name = node.func.id

            # print関数の処理
            if func_name == "print":
                if len(node.args) != 1:
                    raise NotImplementedError("Multiple/zero arguments for print() are not supported")

                # 引数を処理
                arg_typed = self.visit(node.args[0])

                # 文字列化
                str_value = self.get_temp_name()

                # すでに文字列型ならそのまま使用
                if arg_typed.python_type == "str":
                    str_value = arg_typed.llvm_value
                else:
                    # オブジェクトを文字列化（PyObject_Str を使用）
                    obj_value = self.ensure_object(arg_typed)
                    self.builder.emit(f"  {str_value} = call ptr @PyObject_Str(ptr {obj_value.llvm_value})")

                # print関数を呼び出し
                self.builder.emit(f"  call void @print(ptr {str_value})")

                # 戻り値はNone
                tmp = self.get_temp_name()
                self.builder.emit(f"  {tmp} = load ptr, ptr @Py_None")
                self.builder.emit(f"  call void @Py_INCREF(ptr {tmp})")
                return TypedValue.create_object(tmp, "None")

            # str関数の処理
            elif func_name == "str":
                if len(node.args) != 1:
                    raise NotImplementedError("Multiple/zero arguments for str() are not supported")

                # 引数を処理
                arg_typed = self.visit(node.args[0])

                # オブジェクトに変換
                obj_value = self.ensure_object(arg_typed)

                # オブジェクトを文字列化
                tmp = self.get_temp_name()
                self.builder.emit(f"  {tmp} = call ptr @PyObject_Str(ptr {obj_value.llvm_value})")
                return TypedValue.create_object(tmp, "str")

            # その他のユーザー定義関数
            else:
                # 関数シグネチャを取得
                sig = self.get_function_signature(func_name)
                if sig is None:
                    raise NameError(f"Function '{func_name}' not found")

                arg_types, return_type, arg_python_types, return_python_type = sig

                # 引数の数をチェック
                if len(node.args) != len(arg_types):
                    raise TypeError(f"Function '{func_name}' expects {len(arg_types)} arguments, but {len(node.args)} were given")

                # 引数を処理
                processed_args = []

                for i, (arg_node, expected_type, expected_py_type) in enumerate(zip(node.args, arg_types, arg_python_types)):
                    # 引数を評価
                    arg_typed = self.visit(arg_node)

                    # 型を調整
                    if expected_type == "ptr" and expected_py_type != "object":
                        # オブジェクト型が期待される場合
                        arg_value = self.ensure_object(arg_typed)
                    elif arg_typed.type_ != expected_type:
                        # プリミティブ型が期待される場合
                        if arg_typed.is_object:
                            arg_value = self.get_unboxed_value(arg_typed, expected_type)
                        else:
                            # 型変換が必要な場合
                            tmp = self.get_temp_name()
                            self.builder.emit(f"  {tmp} = bitcast {arg_typed.type_} {arg_typed.llvm_value} to {expected_type}")
                            arg_value = TypedValue.create_primitive(tmp, expected_type, arg_typed.python_type)
                    else:
                        # 型が一致する場合
                        arg_value = arg_typed

                    processed_args.append(arg_value)

                # 引数リストを構築
                arg_list = ", ".join(f"{arg.type_} {arg.llvm_value}" for arg in processed_args)

                # 関数呼び出し
                result = self.get_temp_name()
                self.builder.emit(f"  {result} = call {return_type} @{func_name}({arg_list})")

                # 戻り値の型を設定
                if return_type == "ptr" and return_python_type != "object":
                    return TypedValue.create_object(result, return_python_type)
                else:
                    return TypedValue.create_primitive(result, return_type, return_python_type)

        # メソッド呼び出しなど、その他の関数呼び出し形式
        else:
            raise NotImplementedError("Complex function calls are not supported")

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
        定数値を処理する。
        Pythonオブジェクトとして適切に生成する。

        ```asdl
        Constant(constant value, string? kind)
        ```
        """
        val = node.value

        if isinstance(val, int):
            # 整数値は直接LLVMの整数定数として扱う（オブジェクト変換なし）
            return TypedValue.create_primitive(str(val), "i32", "int")

        if isinstance(val, str):
            # 文字列 -> PyUnicodeObject*
            gname = self.builder.add_global_string(val)
            tmp = self.get_temp_name()
            self.builder.emit(f"  {tmp} = call ptr @PyUnicode_FromString(ptr {gname})")
            return TypedValue.create_object(tmp, "str")

        elif isinstance(val, bool):
            # 真偽値 -> Py_True/Py_False
            if val:
                tmp = self.get_temp_name()
                self.builder.emit(f"  {tmp} = load ptr, ptr @Py_True")
                self.builder.emit(f"  call void @Py_INCREF(ptr {tmp})")
                return TypedValue.create_object(tmp, "bool")
            else:
                tmp = self.get_temp_name()
                self.builder.emit(f"  {tmp} = load ptr, ptr @Py_False")
                self.builder.emit(f"  call void @Py_INCREF(ptr {tmp})")
                return TypedValue.create_object(tmp, "bool")

        elif val is None:
            # None -> Py_None
            tmp = self.get_temp_name()
            self.builder.emit(f"  {tmp} = load ptr, ptr @Py_None")
            self.builder.emit(f"  call void @Py_INCREF(ptr {tmp})")
            return TypedValue.create_object(tmp, "None")

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

    def visit_Subscript(self, node: ast.Subscript) -> TypedValue:
        """
        添字アクセスを処理する

        ```asdl
        Subscript(expr value, expr slice, expr_context ctx)
        ```
        """
        # まず基底のオブジェクトの評価
        base_val = self.visit(node.value)

        # オブジェクトを確保
        base_obj = self.ensure_object(base_val)

        # スライスの評価
        if isinstance(node.slice, ast.Constant):
            # 定数インデックス
            const_val = node.slice.value

            if isinstance(const_val, int):
                # 整数インデックス -> リストアクセス
                idx_val = str(const_val)  # 直接数値を使用
                temp = self.builder.get_temp_name()
                self.builder.emit(f"  {temp} = call ptr @PyList_GetItem(ptr {base_obj.llvm_value}, i64 {idx_val})")

                # 参照カウント増加（PyList_GetItemは借用参照を返す）
                self.builder.emit(f"  call void @Py_INCREF(ptr {temp})")

                return TypedValue.create_object(temp, "object")

            elif isinstance(const_val, str):
                # 文字列キー → 辞書アクセス
                key_gname = self.builder.add_global_string(const_val)
                key_obj = self.builder.get_temp_name()
                self.builder.emit(f"  {key_obj} = call ptr @PyUnicode_FromString(ptr {key_gname})")

                # 辞書から取得
                temp = self.builder.get_temp_name()
                self.builder.emit(f"  {temp} = call ptr @PyDict_GetItem(ptr {base_obj.llvm_value}, ptr {key_obj})")

                # キーオブジェクトを解放
                self.builder.emit(f"  call void @Py_DECREF(ptr {key_obj})")

                # NULLチェック（簡略化）
                self.builder.emit("  ; キーが存在するかチェック")

                # 参照カウント増加（借用参照を自前で管理）
                self.builder.emit(f"  call void @Py_INCREF(ptr {temp})")

                return TypedValue.create_object(temp, "object")
            else:
                # その他の定数型はサポート外
                raise TypeError(f"Subscript index of type {type(const_val).__name__} is not supported")
        else:
            # 複雑なスライス式はサポート外
            raise NotImplementedError("Complex slice expressions are not supported")

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
        # True, False, None をオブジェクトとして扱う
        if node.id == "True":
            tmp = self.builder.get_temp_name()
            self.builder.emit(f"  {tmp} = load ptr, ptr @Py_True")
            self.builder.emit(f"  call void @Py_INCREF(ptr {tmp})")
            return TypedValue.create_object(tmp, "bool")

        elif node.id == "False":
            tmp = self.builder.get_temp_name()
            self.builder.emit(f"  {tmp} = load ptr, ptr @Py_False")
            self.builder.emit(f"  call void @Py_INCREF(ptr {tmp})")
            return TypedValue.create_object(tmp, "bool")

        elif node.id == "None":
            tmp = self.builder.get_temp_name()
            self.builder.emit(f"  {tmp} = load ptr, ptr @Py_None")
            self.builder.emit(f"  call void @Py_INCREF(ptr {tmp})")
            return TypedValue.create_object(tmp, "None")

        else:
            # 変数の型情報を取得
            var_type_info = self.get_symbol_type(node.id)
            if var_type_info is None:
                raise NameError(f"Use of variable '{node.id}' before assignment")

            # 型情報を展開
            llvm_type, python_type, is_object = var_type_info

            return TypedValue.create_primitive(f"%{node.id}", llvm_type, python_type)

    def visit_List(self, node: ast.List) -> TypedValue:
        """
        リストリテラルを処理する。
        PyListObject*を生成し、要素を追加する。

        ```asdl
        List(expr* elts, expr_context ctx)
        ```
        """
        # 新しいリストを作成
        size = len(node.elts)
        temp_list = self.get_temp_name()
        self.builder.emit(f"  {temp_list} = call ptr @PyList_New(i64 {size})")

        # 各要素を追加
        for i, elt in enumerate(node.elts):
            # 要素を評価
            elt_typed = self.visit(elt)

            # すべての要素をオブジェクトとして扱う
            elt_obj = self.ensure_object(elt_typed)

            # 参照カウントを増やす（PyList_SetItemは参照を盗まないので）
            self.builder.emit(f"  call void @Py_INCREF(ptr {elt_obj.llvm_value})")

            # リストに追加
            self.builder.emit(f"  call i32 @PyList_SetItem(ptr {temp_list}, i64 {i}, ptr {elt_obj.llvm_value})")

        return TypedValue.create_object(temp_list, "list")

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
