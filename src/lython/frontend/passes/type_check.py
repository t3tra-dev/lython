"""
型チェックパス

TypedASTに対して型チェックを実行し、型不一致をエラーとして報告します。
native部分は厳密、一般Python部分は段階的型付けで柔軟にチェックします。
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any, List, Optional

from ..ast_nodes import (
    Span,
    Ty,
    TypedAnnAssign,
    TypedAssign,
    TypedASTVisitor,
    TypedAttribute,
    TypedAugAssign,
    TypedBinOp,
    TypedCall,
    TypedClassDef,
    TypedConstant,
    TypedExpr,
    TypedFor,
    TypedFunctionDef,
    TypedIf,
    TypedModule,
    TypedName,
    TypedReturn,
    TypedSubscript,
    TypedUnaryOp,
    TypedWhile,
)
from ..typing.error_reporter import ErrorReporter
from ..typing.solver import SymbolTable, TypeInferenceEngine
from ..typing.types import (
    AnyType,
    FunctionType,
    GenericType,
    PrimitiveType,
    PyObjectType,
    Type,
    UnknownType,
    get_common_supertype,
)


@dataclass
class TypeCheckContext:
    """型チェックのコンテキスト情報"""

    symbol_table: SymbolTable
    current_function: Optional[TypedFunctionDef] = None
    in_native_scope: bool = False
    strict_mode: bool = False  # --strict フラグ
    allow_implicit_conversions: bool = True

    def is_native_context(self) -> bool:
        """現在のコンテキストがnativeかどうか"""
        if self.in_native_scope:
            return True
        if self.current_function and self.current_function.is_native:
            return True
        return False


class TypeCheckPass(TypedASTVisitor):
    """型チェックを実行するパス"""

    def __init__(self, error_reporter: Optional[ErrorReporter] = None) -> None:
        self.reporter = error_reporter or ErrorReporter()
        self.context = TypeCheckContext(SymbolTable())
        self.inference_engine = TypeInferenceEngine()

        # 組み込み関数の型を登録
        self._register_builtin_functions()

    def run(self, module: TypedModule, strict_mode: bool = False) -> TypedModule:
        """型チェックパスを実行"""
        self.context.strict_mode = strict_mode

        # 型推論を先に実行
        module = self.inference_engine.infer(module)

        # エラーがあれば報告に追加
        for error in self.inference_engine.errors:
            # エラーメッセージを解析してSpanを抽出（簡略化）
            span = Span(0, 0)  # 実際にはエラーから位置情報を抽出
            self.reporter.add_error(error, span)

        # 型チェックを実行
        self.visit(module)

        return module

    def _register_builtin_functions(self) -> None:
        """組み込み関数の型をシンボルテーブルに登録"""
        # print関数
        print_type = FunctionType([AnyType()], PyObjectType.none())
        self.context.symbol_table.define("print", print_type)

        # len関数
        len_type = FunctionType([AnyType()], PyObjectType.int())
        self.context.symbol_table.define("len", len_type)

        # str, int, float, bool 型変換関数
        self.context.symbol_table.define(
            "str", FunctionType([AnyType()], PyObjectType.str())
        )
        self.context.symbol_table.define(
            "int", FunctionType([AnyType()], PyObjectType.int())
        )
        self.context.symbol_table.define(
            "float", FunctionType([AnyType()], PyObjectType.float())
        )
        self.context.symbol_table.define(
            "bool", FunctionType([AnyType()], PyObjectType.bool())
        )

        # type関数
        self.context.symbol_table.define("type", FunctionType([AnyType()], AnyType()))

        # isinstance関数
        self.context.symbol_table.define(
            "isinstance", FunctionType([AnyType(), AnyType()], PyObjectType.bool())
        )

    def visit_module(self, node: TypedModule) -> TypedModule:
        """モジュール全体の型チェック"""
        for stmt in node.body:
            self.visit(stmt)
        return node

    def visit_function_def(self, node: TypedFunctionDef) -> TypedFunctionDef:
        """関数定義の型チェック"""
        # 前のコンテキストを保存
        old_function = self.context.current_function
        old_scope = self.context.in_native_scope
        old_table = self.context.symbol_table

        # 新しいコンテキストを設定
        self.context.current_function = node
        self.context.in_native_scope = node.is_native
        self.context.symbol_table = self.context.symbol_table.create_child(
            node.is_native
        )

        try:
            # 引数の型チェック
            self._check_function_arguments(node)

            # 戻り値型のチェック
            if node.returns:
                return_type = self._convert_type_annotation(node.returns)
                if node.is_native and isinstance(return_type, UnknownType):
                    self.reporter.add_native_annotation_required_error(
                        node.name, node.span, "return type"
                    )
            else:
                if node.is_native:
                    self.reporter.add_native_annotation_required_error(
                        node.name, node.span, "return type"
                    )

            # 関数本体の型チェック
            for stmt in node.body:
                self.visit(stmt)

            # return文の戻り値型チェック
            self._check_return_statements(node)

        finally:
            # コンテキストを復元
            self.context.current_function = old_function
            self.context.in_native_scope = old_scope
            self.context.symbol_table = old_table

        return node

    def visit_class_def(self, node: TypedClassDef) -> TypedClassDef:
        """クラス定義の型チェック"""
        # 新しいスコープを作成
        old_table = self.context.symbol_table
        self.context.symbol_table = self.context.symbol_table.create_child()

        try:
            # クラス本体の型チェック
            for stmt in node.body:
                self.visit(stmt)
        finally:
            self.context.symbol_table = old_table

        return node

    def visit_assign(self, node: TypedAssign) -> TypedAssign:
        """代入文の型チェック"""
        # 右辺の型チェック
        self.visit(node.value)
        value_type = self._get_type_from_node(node.value)

        # 左辺の各ターゲットをチェック
        for target in node.targets:
            self.visit(target)
            self._check_assignment_compatibility(target, value_type, node.span)

        return node

    def visit_ann_assign(self, node: TypedAnnAssign) -> TypedAnnAssign:
        """型注釈付き代入の型チェック"""
        # 型注釈の型を取得
        annotation_type = self._convert_type_annotation(node.annotation)

        if node.value:
            # 値がある場合は型の互換性をチェック
            self.visit(node.value)
            value_type = self._get_type_from_node(node.value)

            if not self._is_assignable(value_type, annotation_type):
                self.reporter.add_type_error(
                    annotation_type, value_type, node.span, "annotated assignment"
                )

        # ターゲットの型チェック
        self.visit(node.target)

        return node

    def visit_aug_assign(self, node: TypedAugAssign) -> TypedAugAssign:
        """拡張代入の型チェック"""
        # 左辺と右辺の型チェック
        self.visit(node.target)
        self.visit(node.value)

        target_type = self._get_type_from_node(node.target)
        value_type = self._get_type_from_node(node.value)

        # 演算結果の型を推論
        result_type = self._infer_binop_result_type(node.op, target_type, value_type)

        if not self._is_assignable(result_type, target_type):
            self.reporter.add_type_error(
                target_type, result_type, node.span, "augmented assignment"
            )

        return node

    def visit_return(self, node: TypedReturn) -> TypedReturn:
        """return文の型チェック"""
        if not self.context.current_function:
            self.reporter.add_error(
                "return statement outside function", node.span, "E009"
            )
            return node

        # 戻り値の型チェック
        if node.value:
            self.visit(node.value)
            return_type = self._get_type_from_node(node.value)
        else:
            return_type = PyObjectType.none()

        # 関数の戻り値型と比較
        func = self.context.current_function
        if func.returns:
            expected_type = self._convert_type_annotation(func.returns)
            if not self._is_assignable(return_type, expected_type):
                self.reporter.add_return_type_mismatch_error(
                    expected_type, return_type, node.span, func.name
                )

        return node

    def visit_if(self, node: TypedIf) -> TypedIf:
        """if文の型チェック"""
        # 条件式の型チェック
        self.visit(node.test)
        test_type = self._get_type_from_node(node.test)

        # 条件式はbool型に変換可能である必要
        if not self._is_truthy_type(test_type):
            self.reporter.add_warning(
                f"Condition expression of type '{test_type}' may not be truthy",
                node.test.span,
                "W003",
            )

        # if本体とelse部の型チェック
        for stmt in node.body:
            self.visit(stmt)

        for stmt in node.orelse:
            self.visit(stmt)

        return node

    def visit_while(self, node: TypedWhile) -> TypedWhile:
        """while文の型チェック"""
        # 条件式の型チェック
        self.visit(node.test)
        test_type = self._get_type_from_node(node.test)

        if not self._is_truthy_type(test_type):
            self.reporter.add_warning(
                f"Condition expression of type '{test_type}' may not be truthy",
                node.test.span,
                "W003",
            )

        # ループ本体の型チェック
        for stmt in node.body:
            self.visit(stmt)

        for stmt in node.orelse:
            self.visit(stmt)

        return node

    def visit_for(self, node: TypedFor) -> TypedFor:
        """for文の型チェック"""
        # イテラブルの型チェック
        self.visit(node.iter)
        iter_type = self._get_type_from_node(node.iter)

        # イテレータの要素型を推論
        element_type = self._infer_iteration_type(iter_type)

        # ターゲット変数の型チェック
        self.visit(node.target)
        target_type = self._get_type_from_node(node.target)

        if not self._is_assignable(element_type, target_type):
            self.reporter.add_type_error(
                target_type, element_type, node.target.span, "for loop target"
            )

        # ループ本体の型チェック
        for stmt in node.body:
            self.visit(stmt)

        for stmt in node.orelse:
            self.visit(stmt)

        return node

    def visit_binop(self, node: TypedBinOp) -> TypedBinOp:
        """二項演算の型チェック"""
        self.visit(node.left)
        self.visit(node.right)

        left_type = self._get_type_from_node(node.left)
        right_type = self._get_type_from_node(node.right)

        # 演算結果の型を推論
        result_type = self._infer_binop_result_type(node.op, left_type, right_type)

        # native contextでの型チェック
        if self.context.is_native_context():
            if not self._is_native_compatible_operation(node.op, left_type, right_type):
                self.reporter.add_error(
                    f"Operation '{self._get_op_symbol(node.op)}' not supported "
                    f"for native types {left_type} and {right_type}",
                    node.span,
                    "E010",
                )

        # 型をノードに設定
        if hasattr(node, "ty"):
            node.ty = Ty(str(result_type), result_type.get_llvm_type())

        return node

    def visit_unaryop(self, node: TypedUnaryOp) -> TypedUnaryOp:
        """単項演算の型チェック"""
        self.visit(node.operand)
        operand_type = self._get_type_from_node(node.operand)

        # 演算結果の型を推論
        result_type = self._infer_unaryop_result_type(node.op, operand_type)

        if hasattr(node, "ty"):
            node.ty = Ty(str(result_type), result_type.get_llvm_type())

        return node

    def visit_call(self, node: TypedCall) -> TypedCall:
        """関数呼び出しの型チェック"""
        # 関数オブジェクトの型チェック
        self.visit(node.func)
        func_type = self._get_type_from_node(node.func)

        # 引数の型チェック
        arg_types: List[Type] = []
        for arg in node.args:
            self.visit(arg)
            arg_types.append(self._get_type_from_node(arg))

        # 関数型かどうかチェック
        if isinstance(func_type, FunctionType):
            self._check_function_call(node, func_type, arg_types)
            result_type = func_type.return_type
        elif isinstance(func_type, AnyType):
            # Any型の場合は何でも可能
            result_type = AnyType()
        else:
            # 呼び出し不可能
            self.reporter.add_non_callable_error(str(func_type), node.span)
            result_type = AnyType()

        if hasattr(node, "ty"):
            node.ty = Ty(str(result_type), result_type.get_llvm_type())

        return node

    def visit_attribute(self, node: TypedAttribute) -> TypedAttribute:
        """属性参照の型チェック"""
        self.visit(node.value)
        obj_type = self._get_type_from_node(node.value)

        # 属性の存在チェック（簡略化）
        if not self._has_attribute(obj_type, node.attr):
            self.reporter.add_attribute_error(str(obj_type), node.attr, node.span)

        # 属性の型を推論（簡略化）
        attr_type = self._infer_attribute_type(obj_type, node.attr)

        if hasattr(node, "ty"):
            node.ty = Ty(str(attr_type), attr_type.get_llvm_type())

        return node

    def visit_subscript(self, node: TypedSubscript) -> TypedSubscript:
        """添字参照の型チェック"""
        self.visit(node.value)
        self.visit(node.slice)

        obj_type = self._get_type_from_node(node.value)
        slice_type = self._get_type_from_node(node.slice)

        # 添字アクセス可能かチェック
        if not self._is_subscriptable(obj_type):
            self.reporter.add_index_error(str(obj_type), node.span)

        # 要素型を推論
        element_type = self._infer_subscript_type(obj_type, slice_type)

        if hasattr(node, "ty"):
            node.ty = Ty(str(element_type), element_type.get_llvm_type())

        return node

    def visit_name(self, node: TypedName) -> TypedName:
        """変数参照の型チェック"""
        var_type = self.context.symbol_table.lookup(node.id)
        if var_type is None:
            self.reporter.add_undefined_variable_error(node.id, node.span)
            var_type = AnyType()

        if hasattr(node, "ty"):
            node.ty = Ty(str(var_type), var_type.get_llvm_type())

        return node

    def visit_constant(self, node: TypedConstant) -> TypedConstant:
        """定数の型チェック"""
        const_type = self._infer_constant_type(node.value)

        if hasattr(node, "ty"):
            node.ty = Ty(str(const_type), const_type.get_llvm_type())

        return node

    # ヘルパーメソッド

    def _check_function_arguments(self, node: TypedFunctionDef) -> None:
        """関数引数の型チェック"""
        for arg in node.args.args:
            if arg.annotation:
                arg_type = self._convert_type_annotation(arg.annotation)
                if node.is_native and isinstance(arg_type, UnknownType):
                    self.reporter.add_native_annotation_required_error(
                        node.name, node.span, f"parameter '{arg.arg}'"
                    )

                # 引数をシンボルテーブルに登録
                self.context.symbol_table.define(arg.arg, arg_type)
            else:
                if node.is_native:
                    self.reporter.add_native_annotation_required_error(
                        node.name, node.span, f"parameter '{arg.arg}'"
                    )
                # 型注釈がない場合はAny型
                self.context.symbol_table.define(arg.arg, AnyType())

    def _check_return_statements(self, node: TypedFunctionDef) -> None:
        """return文の戻り値型チェック"""
        # この実装は簡略化されています
        # 実際には、関数内のすべてのreturn文を収集して型をチェックする必要があります
        pass

    def _check_assignment_compatibility(
        self, target: TypedExpr, value_type: Type, span: Span
    ) -> None:
        """代入の互換性チェック"""
        if isinstance(target, TypedName):
            # 変数への代入
            existing_type = self.context.symbol_table.lookup(target.id)
            if existing_type:
                if not self._is_assignable(value_type, existing_type):
                    self.reporter.add_type_error(
                        existing_type, value_type, span, "assignment"
                    )
            else:
                # 新しい変数の定義
                self.context.symbol_table.define(target.id, value_type)

    def _check_function_call(
        self, node: TypedCall, func_type: FunctionType, arg_types: List[Type]
    ) -> None:
        """関数呼び出しの型チェック"""
        # 引数の数をチェック
        if len(arg_types) != len(func_type.param_types):
            func_name = ""
            if isinstance(node.func, TypedName):
                func_name = node.func.id
            self.reporter.add_argument_count_error(
                len(func_type.param_types), len(arg_types), node.span, func_name
            )
            return

        # 各引数の型をチェック
        for i, (arg_type, param_type) in enumerate(
            zip(arg_types, func_type.param_types)
        ):
            if not self._is_assignable(arg_type, param_type):
                self.reporter.add_type_error(
                    param_type, arg_type, node.args[i].span, f"argument {i + 1}"
                )

    def _convert_type_annotation(self, annotation: TypedExpr) -> Type:
        """型注釈をType オブジェクトに変換"""
        # この実装は簡略化されています
        if hasattr(annotation, "ty") and annotation.ty:
            type_map = {
                "int": PyObjectType.int(),
                "float": PyObjectType.float(),
                "str": PyObjectType.str(),
                "bool": PyObjectType.bool(),
                "i32": PrimitiveType.i32(),
                "i64": PrimitiveType.i64(),
                "f32": PrimitiveType.f32(),
                "f64": PrimitiveType.f64(),
            }
            return type_map.get(annotation.ty.py_name, AnyType())
        return AnyType()

    def _get_type_from_node(self, node: TypedExpr) -> Type:
        """ノードから型を取得"""
        if hasattr(node, "ty") and node.ty:
            type_map = {
                "int": PyObjectType.int(),
                "float": PyObjectType.float(),
                "str": PyObjectType.str(),
                "bool": PyObjectType.bool(),
                "i32": PrimitiveType.i32(),
                "i64": PrimitiveType.i64(),
                "f32": PrimitiveType.f32(),
                "f64": PrimitiveType.f64(),
            }
            return type_map.get(node.ty.py_name, AnyType())
        return UnknownType()

    def _is_assignable(self, from_type: Type, to_type: Type) -> bool:
        """型の代入可能性をチェック"""
        # strictモードでの厳密チェック
        if self.context.strict_mode:
            return from_type.is_assignable_to(to_type)

        # 段階的型付けでの緩やかなチェック
        if isinstance(to_type, AnyType) or isinstance(from_type, AnyType):
            return True

        if from_type.is_assignable_to(to_type):
            return True

        # 暗黙的変換の許可
        if self.context.allow_implicit_conversions:
            return self._can_implicitly_convert(from_type, to_type)

        return False

    def _can_implicitly_convert(self, from_type: Type, to_type: Type) -> bool:
        """暗黙的変換が可能かどうか"""
        # プリミティブ型間の変換
        if isinstance(from_type, PrimitiveType) and isinstance(to_type, PrimitiveType):
            return from_type.can_convert_to(to_type)

        # PyObject型間の変換
        if isinstance(from_type, PyObjectType) and isinstance(to_type, PyObjectType):
            return from_type.is_subtype_of(to_type)

        # 数値型の相互変換
        numeric_types = {"int", "float", "i32", "i64", "f32", "f64"}
        if str(from_type) in numeric_types and str(to_type) in numeric_types:
            return True

        return False

    def _infer_constant_type(self, value: Any) -> Type:
        """定数値から型を推論"""
        if isinstance(value, int):
            return PyObjectType.int()
        elif isinstance(value, float):
            return PyObjectType.float()
        elif isinstance(value, str):
            return PyObjectType.str()
        elif isinstance(value, bool):
            return PyObjectType.bool()
        elif value is None:
            return PyObjectType.none()
        else:
            return AnyType()

    def _infer_binop_result_type(
        self, op: ast.operator, left: Type, right: Type
    ) -> Type:
        """二項演算の結果型を推論"""
        # 簡略化された実装
        if isinstance(op, ast.Add):
            if str(left) == "str" or str(right) == "str":
                return PyObjectType.str()
            elif (
                hasattr(left, "is_numeric")
                and hasattr(right, "is_numeric")
                and left.is_numeric()
                and right.is_numeric()
            ):
                return get_common_supertype([left, right])
        elif isinstance(op, (ast.Sub, ast.Mult)):
            if (
                hasattr(left, "is_numeric")
                and hasattr(right, "is_numeric")
                and left.is_numeric()
                and right.is_numeric()
            ):
                return get_common_supertype([left, right])
        elif isinstance(op, ast.Div):
            if (
                hasattr(left, "is_numeric")
                and hasattr(right, "is_numeric")
                and left.is_numeric()
                and right.is_numeric()
            ):
                return PyObjectType.float()
        elif isinstance(op, (ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE)):
            return PyObjectType.bool()

        return AnyType()

    def _infer_unaryop_result_type(self, op: ast.unaryop, operand_type: Type) -> Type:
        """単項演算の結果型を推論"""
        if isinstance(op, ast.UAdd) or isinstance(op, ast.USub):
            if hasattr(operand_type, "is_numeric") and operand_type.is_numeric():
                return operand_type
        elif isinstance(op, ast.Not):
            return PyObjectType.bool()

        return AnyType()

    def _is_truthy_type(self, ty: Type) -> bool:
        """型がtruthyかどうか"""
        # ほとんどの型はtruthyと見なす（簡略化）
        return True

    def _infer_iteration_type(self, iter_type: Type) -> Type:
        """イテレータの要素型を推論"""
        if isinstance(iter_type, GenericType):
            if len(iter_type.type_args) > 0:
                return iter_type.type_args[0]
        elif str(iter_type) == "str":
            return PyObjectType.str()  # 文字列の各文字

        return AnyType()

    def _is_native_compatible_operation(
        self, op: ast.operator, left: Type, right: Type
    ) -> bool:
        """native型でサポートされる演算かどうか"""
        if not (left.is_native() and right.is_native()):
            return False

        # プリミティブ型間の基本演算はサポート
        if isinstance(left, PrimitiveType) and isinstance(right, PrimitiveType):
            return True

        return False

    def _get_op_symbol(self, op: ast.operator) -> str:
        """演算子のシンボル文字列を取得"""
        op_symbols: dict[type, str] = {
            ast.Add: "+",
            ast.Sub: "-",
            ast.Mult: "*",
            ast.Div: "/",
            ast.Mod: "%",
            ast.Pow: "**",
            ast.LShift: "<<",
            ast.RShift: ">>",
            ast.BitOr: "|",
            ast.BitXor: "^",
            ast.BitAnd: "&",
            ast.FloorDiv: "//",
            ast.Eq: "==",
            ast.NotEq: "!=",
            ast.Lt: "<",
            ast.LtE: "<=",
            ast.Gt: ">",
            ast.GtE: ">=",
        }
        return op_symbols.get(type(op), str(op))

    def _has_attribute(self, obj_type: Type, attr: str) -> bool:
        """オブジェクトが指定された属性を持つかどうか"""
        # 簡略化された実装（実際にはより詳細な属性解析が必要）
        return True

    def _infer_attribute_type(self, obj_type: Type, attr: str) -> Type:
        """属性の型を推論"""
        # 簡略化された実装
        return AnyType()

    def _is_subscriptable(self, obj_type: Type) -> bool:
        """オブジェクトが添字アクセス可能かどうか"""
        subscriptable_types = {"list", "dict", "tuple", "str"}
        return str(obj_type) in subscriptable_types or isinstance(obj_type, GenericType)

    def _infer_subscript_type(self, obj_type: Type, slice_type: Type) -> Type:
        """添字アクセスの結果型を推論"""
        if isinstance(obj_type, GenericType):
            if len(obj_type.type_args) > 0:
                return obj_type.type_args[0]
        elif str(obj_type) == "str":
            return PyObjectType.str()

        return AnyType()
