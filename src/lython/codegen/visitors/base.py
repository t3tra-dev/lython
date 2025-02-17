import ast
from typing import Any, Dict, List, Optional, Tuple

from ..ir import IRBuilder

__all__ = ["BaseVisitor", "TypedValue"]


class TypedValue:
    """
    各式を評価した結果:
      - llvm_value: IR上の値 (例: '%t0' / '42' / 'null')
      - type_: 'i32', 'i1', 'ptr' など
    """
    def __init__(self, llvm_value: str, type_: str):
        self.llvm_value = llvm_value
        self.type_ = type_


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

        # 簡易的なシンボルテーブル（変数名 -> 型など）
        self.symbol_table: Dict[str, str] = {}
        # 関数の引数と戻り値の型情報
        self.function_signatures: Dict[str, Tuple[List[str], str]] = {}
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

    def set_symbol_type(self, name: str, t: str):
        self.symbol_table[name] = t

    def get_symbol_type(self, name: str) -> str | None:
        return self.symbol_table.get(name)

    def set_function_signature(self, name: str, arg_types: List[str], return_type: str):
        self.function_signatures[name] = (arg_types, return_type)

    def get_function_signature(self, name: str) -> Optional[Tuple[List[str], str]]:
        return self.function_signatures.get(name)

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
