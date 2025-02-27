import ast
from typing import Any, Dict, List, Optional, Tuple

from ..ir import IRBuilder

__all__ = ["BaseVisitor", "TypedValue"]


class TypedValue:
    """
    各式を評価した結果を表現するクラス

    Attributes:
        llvm_value (str): IR上の値 (例: '%t0' / '42' / 'null')
        type_ (str): LLVM IR上の型 (例: 'i32', 'i1', 'ptr')
        python_type (str): Python上の型名 (例: 'int', 'str', 'list')
        is_object (bool): PyObjectの派生型かどうか
    """
    def __init__(
        self,
        llvm_value: str,
        type_: str,
        python_type: str = "object",
        is_object: bool = False
    ):
        self.llvm_value = llvm_value
        self.type_ = type_
        self.python_type = python_type
        self.is_object = is_object

    def __repr__(self) -> str:
        return f"TypedValue({self.llvm_value}, {self.type_}, {self.python_type}, {self.is_object})"

    @classmethod
    def create_primitive(cls, llvm_value: str, type_: str, python_type: str) -> 'TypedValue':
        """プリミティブ型の値を生成"""
        return cls(llvm_value, type_, python_type, False)

    @classmethod
    def create_object(cls, llvm_value: str, python_type: str) -> 'TypedValue':
        """オブジェクト型の値を生成"""
        return cls(llvm_value, "ptr", python_type, True)

    def needs_boxing(self) -> bool:
        """この値がボクシングが必要かどうかを返す"""
        return not self.is_object and self.python_type in ("int", "float", "bool")

    def needs_unboxing(self) -> bool:
        """この値がアンボクシングが必要かどうかを返す"""
        return self.is_object and self.python_type in ("int", "float", "bool")


class BaseVisitor:
    """
    ベースとなるVisitorクラス

    同時に、簡易的なシンボルテーブルを持ち、
      - 変数名 -> 型
    のマッピングを扱う。
    すべてのASTノードに対して visit_<NodeType>() メソッドをディスパッチする
    また、専用のサブビジター (AliasVisitor, ArgVisitor, etc. など) が存在する場合は、
    self.subvisitors に登録されている対応するVisitorへ転送。
    """

    def __init__(self, builder: IRBuilder):
        self.builder = builder

        # 簡易的なシンボルテーブル
        # 変数名 -> (LLVM型, Python型, is_object)
        self.symbol_table: Dict[str, Tuple[str, str, bool]] = {}

        # 関数の引数と戻り値の型情報
        # 関数名 -> ([引数型リスト], 戻り値型, [引数Python型リスト], 戻り値Python型)
        self.function_signatures: Dict[str, Tuple[List[str], str, List[str], str]] = {}

        # サブビジターの辞書。各キーは ASTノードのクラス名
        self.subvisitors: Dict[str, BaseVisitor] = {}

    def _init_subvisitors(self):
        from .alias import AliasVisitor
        from .arg import ArgVisitor
        from .arguments import ArgumentsVisitor
        from .boolop import BoolOpVisitor
        from .cmpop import CmpOpVisitor
        from .comprehension import ComprehensionVisitor
        from .excepthandler import ExceptHandlerVisitor
        from .expr_context import ExprContextVisitor
        from .expr import ExprVisitor
        from .keyword import KeywordVisitor
        from .match_case import MatchCaseVisitor
        from .mod import ModVisitor
        from .operator import OperatorVisitor
        from .pattern import PatternVisitor
        from .stmt import StmtVisitor
        from .type_ignore import TypeIgnoreVisitor
        from .type_param import TypeParamVisitor
        from .unaryop import UnaryOpVisitor
        from .withitem import WithitemVisitor

        self.subvisitors = {
            "alias": AliasVisitor(self.builder),
            "arg": ArgVisitor(self.builder),
            "arguments": ArgumentsVisitor(self.builder),
            "boolop": BoolOpVisitor(self.builder),
            "cmpop": CmpOpVisitor(self.builder),
            "comprehension": ComprehensionVisitor(self.builder),
            "exceptHandler": ExceptHandlerVisitor(self.builder),
            "expr_context": ExprContextVisitor(self.builder),
            "expr": ExprVisitor(self.builder),
            "keyword": KeywordVisitor(self.builder),
            "match_case": MatchCaseVisitor(self.builder),
            "mod": ModVisitor(self.builder),
            "operator": OperatorVisitor(self.builder),
            "pattern": PatternVisitor(self.builder),
            "stmt": StmtVisitor(self.builder),
            "type_ignore": TypeIgnoreVisitor(self.builder),
            "type_param": TypeParamVisitor(self.builder),
            "unaryop": UnaryOpVisitor(self.builder),
            "withitem": WithitemVisitor(self.builder),
        }

    def get_subvisitor(self, name: str) -> Any:
        if not self.subvisitors:
            self._init_subvisitors()
        return self.subvisitors[name]

    def set_symbol_type(self, name: str, llvm_type: str, python_type: str = "object", is_object: bool = False):
        """変数のシンボルテーブル情報を設定"""
        self.symbol_table[name] = (llvm_type, python_type, is_object)

    def get_symbol_type(self, name: str) -> Optional[Tuple[str, str, bool]]:
        """変数の型情報を取得"""
        return self.symbol_table.get(name)

    def set_function_signature(
        self, name: str,
        arg_types: List[str],
        return_type: str,
        arg_python_types: Optional[List[str]] = None,
        return_python_type: str = "object"
    ):
        """関数のシグネチャを設定"""
        if arg_python_types is None:
            arg_python_types = ["object"] * len(arg_types)
        self.function_signatures[name] = (arg_types, return_type, arg_python_types, return_python_type)

    def get_function_signature(self, name: str) -> Optional[Tuple[List[str], str, List[str], str]]:
        """関数のシグネチャを取得"""
        return self.function_signatures.get(name)

    def get_boxed_value(self, typed_value: TypedValue) -> TypedValue:
        """プリミティブ型をボクシングする"""
        if not typed_value.needs_boxing():
            return typed_value

        if typed_value.python_type == "int":
            # int -> PyInt_FromI32
            temp = self.builder.get_temp_name()
            self.builder.emit(f"  {temp} = call ptr @PyInt_FromI32(i32 {typed_value.llvm_value})")
            return TypedValue.create_object(temp, "int")
        elif typed_value.python_type == "bool":
            # bool -> PyBool_FromLong
            temp = self.builder.get_temp_name()
            self.builder.emit(f"  {temp} = call ptr @PyBool_FromLong(i32 {typed_value.llvm_value})")
            return TypedValue.create_object(temp, "bool")

        # その他のプリミティブ型（未実装）
        return typed_value

    def get_unboxed_value(self, typed_value: TypedValue, target_type: str) -> TypedValue:
        """オブジェクト型をアンボクシングする"""
        if not typed_value.needs_unboxing():
            return typed_value

        if typed_value.python_type == "int" and target_type == "i32":
            # PyInt -> i32
            temp = self.builder.get_temp_name()
            self.builder.emit(f"  {temp} = call i32 @PyInt_AsI32(ptr {typed_value.llvm_value})")
            return TypedValue.create_primitive(temp, "i32", "int")
        elif typed_value.python_type == "bool" and target_type == "i1":
            # PyBool -> i1
            temp = self.builder.get_temp_name()
            self.builder.emit(f"  {temp} = call i32 @PyObject_IsTrue(ptr {typed_value.llvm_value})")
            bool_temp = self.builder.get_temp_name()
            self.builder.emit(f"  {bool_temp} = icmp ne i32 {temp}, 0")
            return TypedValue.create_primitive(bool_temp, "i1", "bool")

        # その他のオブジェクト型（未実装）
        return typed_value

    def ensure_object(self, typed_value: TypedValue) -> TypedValue:
        """値がオブジェクトであることを保証(必要に応じてボックス化)"""
        if typed_value.is_object:
            return typed_value
        return self.get_boxed_value(typed_value)

    def visit(self, node: ast.AST | None) -> Any:
        if node is None:
            return None
        method_name = f"visit_{type(node).__name__}"
        method = getattr(self, method_name, self.generic_visit)
        return method(node)

    def generic_visit(self, node: ast.AST) -> None:
        """
        各ノードに対応する visit_* が未定義の場合、ここにフォールバック
        未実装の構文要素があればエラーを出す
        """
        raise NotImplementedError(f"Node type {type(node).__name__} not implemented by {self.__class__.__name__}")
