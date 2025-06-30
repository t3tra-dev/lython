"""
型推論エンジン

Hindley-Milner風の型推論アルゴリズムを実装します。
段階的型付けをサポートし、native部分は厳密、Python部分は緩やかな型チェックを行います。
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from ..ast_nodes import (
    Span,
    Ty,
    TypedAnnAssign,
    TypedAssign,
    TypedBinOp,
    TypedCall,
    TypedConstant,
    TypedExpr,
    TypedFunctionDef,
    TypedModule,
    TypedName,
)
from .types import (
    AnyType,
    FunctionType,
    GenericType,
    PrimitiveType,
    PyObjectType,
    TupleType,
    Type,
    UnionType,
    UnknownType,
)


@dataclass
class TypeConstraint:
    """型制約を表すクラス"""

    left: Type
    right: Type
    span: Span
    reason: str  # 制約が生成された理由

    def __str__(self) -> str:
        return f"{self.left} ~ {self.right} ({self.reason} at {self.span})"


def _make_constraint_list() -> List["TypeConstraint"]:
    return []


@dataclass
class TypeVariable:
    """型変数"""

    name: str
    constraints: List["TypeConstraint"] = field(default_factory=_make_constraint_list)
    resolved_type: Optional[Type] = None

    def __str__(self) -> str:
        if self.resolved_type:
            return f"${self.name}={self.resolved_type}"
        return f"${self.name}"

    def is_resolved(self) -> bool:
        return self.resolved_type is not None


def _make_symbol_dict() -> Dict[str, Type]:
    return {}


@dataclass
class SymbolTable:
    """シンボルテーブル - 変数名と型の対応を管理"""

    symbols: Dict[str, Type] = field(default_factory=_make_symbol_dict)
    parent: Optional["SymbolTable"] = None
    is_native_scope: bool = False  # @native関数内かどうか

    def lookup(self, name: str) -> Optional[Type]:
        """変数の型を検索"""
        if name in self.symbols:
            return self.symbols[name]
        if self.parent:
            return self.parent.lookup(name)
        return None

    def define(self, name: str, ty: Type) -> None:
        """変数を定義"""
        self.symbols[name] = ty

    def update(self, name: str, ty: Type) -> bool:
        """既存の変数の型を更新"""
        if name in self.symbols:
            self.symbols[name] = ty
            return True
        if self.parent:
            return self.parent.update(name, ty)
        return False

    def create_child(self, is_native: bool = False) -> SymbolTable:
        """子スコープを作成"""
        return SymbolTable(parent=self, is_native_scope=is_native)


class ConstraintSolver:
    """型制約ソルバー - Hindley-Milner風の単一化アルゴリズム"""

    def __init__(self) -> None:
        self.constraints: List[TypeConstraint] = []
        self.type_variables: Dict[str, TypeVariable] = {}
        self.next_var_id = 0

    def fresh_type_var(self, prefix: str = "T") -> TypeVariable:
        """新しい型変数を生成"""
        var_name = f"{prefix}{self.next_var_id}"
        self.next_var_id += 1
        var = TypeVariable(var_name)
        self.type_variables[var_name] = var
        return var

    def add_constraint(self, left: Type, right: Type, span: Span, reason: str) -> None:
        """制約を追加"""
        constraint = TypeConstraint(left, right, span, reason)
        self.constraints.append(constraint)

    def unify(self, t1: Type, t2: Type) -> Optional[Type]:
        """2つの型を単一化"""
        # 同じ型の場合
        if t1 == t2:
            return t1

        # Any型との単一化
        if isinstance(t1, AnyType):
            return t2
        if isinstance(t2, AnyType):
            return t1

        # 未知の型との単一化
        if isinstance(t1, UnknownType):
            return t2
        if isinstance(t2, UnknownType):
            return t1

        # プリミティブ型の単一化
        if isinstance(t1, PrimitiveType) and isinstance(t2, PrimitiveType):
            return self._unify_primitives(t1, t2)

        # PyObject型の単一化
        if isinstance(t1, PyObjectType) and isinstance(t2, PyObjectType):
            return self._unify_pyobjects(t1, t2)

        # プリミティブ型とPyObject型の混在
        if isinstance(t1, PrimitiveType) and isinstance(t2, PyObjectType):
            return self._unify_primitive_pyobject(t1, t2)
        if isinstance(t2, PrimitiveType) and isinstance(t1, PyObjectType):
            return self._unify_primitive_pyobject(t2, t1)

        # Union型の単一化
        if isinstance(t1, UnionType) or isinstance(t2, UnionType):
            return self._unify_unions(t1, t2)

        # 関数型の単一化
        if isinstance(t1, FunctionType) and isinstance(t2, FunctionType):
            return self._unify_functions(t1, t2)

        # タプル型の単一化
        if isinstance(t1, TupleType) and isinstance(t2, TupleType):
            return self._unify_tuples(t1, t2)

        # ジェネリック型の単一化
        if isinstance(t1, GenericType) and isinstance(t2, GenericType):
            return self._unify_generics(t1, t2)

        # 単一化できない場合はUnion型を作成
        return UnionType([t1, t2])

    def _unify_primitives(self, t1: PrimitiveType, t2: PrimitiveType) -> Optional[Type]:
        """プリミティブ型の単一化"""
        if t1 == t2:
            return t1

        # 数値型の暗黙的変換
        if t1.is_numeric() and t2.is_numeric():
            # より広い型に統合
            if t1.can_convert_to(t2):
                return t2
            elif t2.can_convert_to(t1):
                return t1

        return None

    def _unify_pyobjects(self, t1: PyObjectType, t2: PyObjectType) -> Optional[Type]:
        """PyObject型の単一化"""
        if t1 == t2:
            return t1

        # 継承関係を考慮
        if t1.is_subtype_of(t2):
            return t2
        elif t2.is_subtype_of(t1):
            return t1

        # 共通の親型を探す
        if t1.name in ["int", "float"] and t2.name in ["int", "float"]:
            return PyObjectType("float")  # 数値型の統合

        return PyObjectType("object")  # デフォルトでobject型

    def _unify_primitive_pyobject(
        self, prim: PrimitiveType, pyobj: PyObjectType
    ) -> Optional[Type]:
        """プリミティブ型とPyObject型の統合"""
        # 対応するPyObject型に変換
        primitive_to_pyobject = {
            "i32": "int",
            "i64": "int",
            "f32": "float",
            "f64": "float",
            "bool": "bool",
        }

        if prim.name in primitive_to_pyobject:
            expected_pyobj = primitive_to_pyobject[prim.name]
            if pyobj.name == expected_pyobj:
                return pyobj  # PyObject型を優先

        return pyobj  # デフォルトでPyObject型

    def _unify_unions(self, t1: Type, t2: Type) -> Type:
        """Union型の単一化"""
        types: Set[Type] = set()

        if isinstance(t1, UnionType):
            types.update(t1.types)
        else:
            types.add(t1)

        if isinstance(t2, UnionType):
            types.update(t2.types)
        else:
            types.add(t2)

        return UnionType(types)

    def _unify_functions(self, t1: FunctionType, t2: FunctionType) -> Optional[Type]:
        """関数型の単一化"""
        if len(t1.param_types) != len(t2.param_types):
            return None

        # パラメータ型の単一化（反変性）
        unified_params: List[Type] = []
        for p1, p2 in zip(t1.param_types, t2.param_types):
            unified = self.unify(p1, p2)
            if unified is None:
                return None
            unified_params.append(unified)

        # 戻り値型の単一化（共変性）
        unified_return = self.unify(t1.return_type, t2.return_type)
        if unified_return is None:
            return None

        # nativeの一致が必要
        if t1.is_native_func != t2.is_native_func:
            return None

        return FunctionType(unified_params, unified_return, t1.is_native_func)

    def _unify_tuples(self, t1: TupleType, t2: TupleType) -> Optional[Type]:
        """タプル型の単一化"""
        if len(t1.element_types) != len(t2.element_types):
            return None

        unified_elements: List[Type] = []
        for e1, e2 in zip(t1.element_types, t2.element_types):
            unified = self.unify(e1, e2)
            if unified is None:
                return None
            unified_elements.append(unified)

        return TupleType(unified_elements)

    def _unify_generics(self, t1: GenericType, t2: GenericType) -> Optional[Type]:
        """ジェネリック型の単一化"""
        if not self.unify(t1.base, t2.base):
            return None

        if len(t1.type_args) != len(t2.type_args):
            return None

        unified_args: List[Type] = []
        for a1, a2 in zip(t1.type_args, t2.type_args):
            unified = self.unify(a1, a2)
            if unified is None:
                return None
            unified_args.append(unified)

        return GenericType(t1.base, unified_args)

    def solve_constraints(self) -> Dict[str, Type]:
        """制約を解決して型変数の値を決定"""
        # 簡単な制約解決アルゴリズム
        changed = True
        while changed:
            changed = False
            for constraint in self.constraints:
                unified = self.unify(constraint.left, constraint.right)
                if unified and unified != constraint.left:
                    # 型変数の更新などを処理
                    changed = True

        # 型変数の解決結果を返す
        result: Dict[str, Type] = {}
        for var_name, var in self.type_variables.items():
            if var.resolved_type:
                result[var_name] = var.resolved_type
            else:
                result[var_name] = AnyType()  # 未解決の場合はAny

        return result


class TypeInferenceEngine:
    """型推論の主要エンジン"""

    def __init__(self) -> None:
        self.solver = ConstraintSolver()
        self.symbol_table = SymbolTable()
        self.current_function: Optional[TypedFunctionDef] = None
        self.errors: List[str] = []

    def infer(self, module: TypedModule) -> TypedModule:
        """モジュール全体の型推論を実行"""
        # 1. 収集フェーズ: 型制約を収集
        self._collect_constraints(module)

        # 2. 解決フェーズ: 制約を解決
        resolved_types = self.solver.solve_constraints()

        # 3. 型の適用フェーズ: 解決された型をASTに適用
        return self._apply_types(module, resolved_types)

    def _collect_constraints(self, node: Any) -> None:
        """型制約を収集"""
        if isinstance(node, TypedModule):
            for stmt in node.body:
                self._collect_constraints(stmt)

        elif isinstance(node, TypedFunctionDef):
            self._collect_function_constraints(node)

        elif isinstance(node, TypedAssign):
            self._collect_assign_constraints(node)

        elif isinstance(node, TypedAnnAssign):
            self._collect_ann_assign_constraints(node)

        elif isinstance(node, TypedBinOp):
            self._collect_binop_constraints(node)

        elif isinstance(node, TypedCall):
            self._collect_call_constraints(node)

        elif isinstance(node, TypedConstant):
            self._collect_constant_constraints(node)

        elif isinstance(node, TypedName):
            self._collect_name_constraints(node)

        # 他のノード型についても同様に処理

    def _collect_function_constraints(self, node: TypedFunctionDef) -> None:
        """関数定義の型制約を収集"""
        # @nativeデコレータの検出
        is_native = self._is_native_function(node)
        node.is_native = is_native

        # 新しいスコープを作成
        old_table = self.symbol_table
        self.symbol_table = self.symbol_table.create_child(is_native)
        old_function = self.current_function
        self.current_function = node

        try:
            # 引数の型を登録
            param_types: List[Type] = []
            for arg in node.args.args:
                if arg.annotation:
                    arg_type = self._infer_type_annotation(arg.annotation)
                    # arg.ty = arg_type  # Type assignment
                    param_types.append(arg_type)
                    self.symbol_table.define(arg.arg, arg_type)
                else:
                    if is_native:
                        self.errors.append(
                            f"Argument {arg.arg} of @native function requires a type annotation at {node.span}"
                        )
                    arg_type = UnknownType()
                    # arg.ty = arg_type  # Type assignment
                    param_types.append(arg_type)
                    self.symbol_table.define(arg.arg, arg_type)

            # 戻り値の型
            if node.returns:
                return_type = self._infer_type_annotation(node.returns)
            else:
                if is_native:
                    self.errors.append(
                        f"@native function {node.name} requires a return type annotation at "
                        f"{node.span}"
                    )
                return_type = UnknownType()

            # 関数本体の型推論
            for stmt in node.body:
                self._collect_constraints(stmt)

            # 関数型を作成
            func_type = FunctionType(param_types, return_type, is_native)
            node.ty = Ty(str(func_type), func_type.get_llvm_type())

            # 関数名をシンボルテーブルに登録
            old_table.define(node.name, func_type)

        finally:
            self.symbol_table = old_table
            self.current_function = old_function

    def _collect_assign_constraints(self, node: TypedAssign) -> None:
        """代入文の型制約を収集"""
        # 右辺の型推論
        self._collect_constraints(node.value)
        value_type = self._get_expression_type(node.value)

        # 左辺の各ターゲットに型を適用
        for target in node.targets:
            if isinstance(target, TypedName):
                # 変数への代入
                existing_type = self.symbol_table.lookup(target.id)
                if existing_type:
                    # 既存の変数の場合、型の互換性をチェック
                    unified = self.solver.unify(existing_type, value_type)
                    if unified:
                        self.symbol_table.update(target.id, unified)
                        target.ty = Ty(str(unified), unified.get_llvm_type())
                    else:
                        self.errors.append(
                            f"Type mismatch: {target.id} is expected to be {existing_type}, "
                            f"but {value_type} was assigned at {node.span}"
                        )
                else:
                    # 新しい変数の定義
                    self.symbol_table.define(target.id, value_type)
                    target.ty = Ty(str(value_type), value_type.get_llvm_type())

    def _collect_ann_assign_constraints(self, node: TypedAnnAssign) -> None:
        """型注釈付き代入の型制約を収集"""
        # 型注釈から型を取得
        annotation_type = self._infer_type_annotation(node.annotation)

        if node.value:
            # 値がある場合は型の互換性をチェック
            self._collect_constraints(node.value)
            value_type = self._get_expression_type(node.value)

            if not value_type.is_assignable_to(annotation_type):
                self.errors.append(
                    f"Type mismatch: Expected {annotation_type}, but got "
                    f"{value_type} at {node.span}"
                )

        # 変数をシンボルテーブルに登録
        if isinstance(node.target, TypedName):
            self.symbol_table.define(node.target.id, annotation_type)
            node.target.ty = Ty(str(annotation_type), annotation_type.get_llvm_type())

    def _collect_binop_constraints(self, node: TypedBinOp) -> None:
        """二項演算の型制約を収集"""
        self._collect_constraints(node.left)
        self._collect_constraints(node.right)

        left_type = self._get_expression_type(node.left)
        right_type = self._get_expression_type(node.right)

        # 演算子に基づいて結果の型を決定
        result_type = self._infer_binop_type(node.op, left_type, right_type, node.span)
        node.ty = Ty(str(result_type), result_type.get_llvm_type())

    def _collect_call_constraints(self, node: TypedCall) -> None:
        """関数呼び出しの型制約を収集"""
        self._collect_constraints(node.func)

        for arg in node.args:
            self._collect_constraints(arg)

        func_type = self._get_expression_type(node.func)

        if isinstance(func_type, FunctionType):
            # 引数の数と型をチェック
            if len(node.args) != len(func_type.param_types):
                self.errors.append(
                    f"Invalid number of arguments: Expected {len(func_type.param_types)}, "
                    f"but got {len(node.args)} at {node.span}"
                )
            else:
                for i, (arg, expected_type) in enumerate(
                    zip(node.args, func_type.param_types)
                ):
                    arg_type = self._get_expression_type(arg)
                    if not arg_type.is_assignable_to(expected_type):
                        self.errors.append(
                            f"Type mismatch for argument {i + 1}: Expected {expected_type}, "
                            f"but got {arg_type} at {node.span}"
                        )

            node.ty = Ty(
                str(func_type.return_type), func_type.return_type.get_llvm_type()
            )
        else:
            # 関数型でない場合はエラー
            self.errors.append(f"呼び出し不可能な型: {func_type} at {node.span}")
            node.ty = Ty("Any", "%pyobj*")

    def _collect_constant_constraints(self, node: TypedConstant) -> None:
        """定数の型制約を収集"""
        const_type = self._infer_constant_type(node.value)
        node.ty = Ty(str(const_type), const_type.get_llvm_type())

    def _collect_name_constraints(self, node: TypedName) -> None:
        """変数参照の型制約を収集"""
        var_type = self.symbol_table.lookup(node.id)
        if var_type:
            node.ty = Ty(str(var_type), var_type.get_llvm_type())
        else:
            self.errors.append(f"Undefined variable: {node.id} at {node.span}")
            node.ty = Ty("Any", "%pyobj*")

    def _is_native_function(self, node: TypedFunctionDef) -> bool:
        """@nativeデコレータが付いているかチェック"""
        for decorator in node.decorator_list:
            if isinstance(decorator, TypedName) and decorator.id == "native":
                return True
            # elif isinstance(decorator, TypedAttribute):
            # native.native のような形式
            # Commented out due to import issues
        return False

    def _infer_type_annotation(self, annotation: TypedExpr) -> Type:
        """型注釈から型を推論"""
        if isinstance(annotation, TypedName):
            # 基本型名
            type_map = {
                "int": PyObjectType.int(),
                "float": PyObjectType.float(),
                "str": PyObjectType.str(),
                "bool": PyObjectType.bool(),
                "i32": PrimitiveType.i32(),
                "i64": PrimitiveType.i64(),
                "f32": PrimitiveType.f32(),
                "f64": PrimitiveType.f64(),
                "Any": AnyType(),
            }
            return type_map.get(annotation.id, AnyType())

        # elif isinstance(annotation, TypedSubscript):
        # ジェネリック型 List[int], Dict[str, int] など
        # Commented out due to import issues

        return AnyType()

    def _get_expression_type(self, expr: TypedExpr) -> Type:
        """式の型を取得"""
        if hasattr(expr, "ty") and expr.ty:
            # 基本型マッピング
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
            return type_map.get(expr.ty.py_name, AnyType())
        return UnknownType()

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

    def _infer_binop_type(
        self, op: ast.operator, left: Type, right: Type, span: Span
    ) -> Type:
        """二項演算の結果型を推論"""
        # 加算
        if isinstance(op, ast.Add):
            if isinstance(left, PrimitiveType) and isinstance(right, PrimitiveType):
                return self.solver.unify(left, right) or UnionType([left, right])
            elif left.is_assignable_to(PyObjectType.int()) and right.is_assignable_to(
                PyObjectType.int()
            ):
                return PyObjectType.int()
            elif left.is_assignable_to(PyObjectType.str()) or right.is_assignable_to(
                PyObjectType.str()
            ):
                return PyObjectType.str()
            else:
                return self.solver.unify(left, right) or AnyType()

        # 他の演算子も同様に実装
        elif isinstance(op, (ast.Sub, ast.Mult, ast.Div)):
            if isinstance(left, PrimitiveType) and isinstance(right, PrimitiveType):
                return self.solver.unify(left, right) or UnionType([left, right])
            elif left.is_assignable_to(PyObjectType.int()) and right.is_assignable_to(
                PyObjectType.int()
            ):
                if isinstance(op, ast.Div):
                    return PyObjectType.float()  # 除算は常にfloat
                return PyObjectType.int()
            else:
                return AnyType()

        # 比較演算
        elif isinstance(op, (ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE)):
            return PyObjectType.bool()

        return AnyType()

    def _apply_types(
        self, module: TypedModule, resolved_types: Dict[str, Type]
    ) -> TypedModule:
        """解決された型をASTに適用"""
        # 型変数を解決された型で置き換える処理
        # 実装は複雑になるため、ここでは基本的な処理のみ
        return module
