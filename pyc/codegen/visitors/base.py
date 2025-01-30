import ast
from typing import Any

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
    すべてのノードタイプに対して visit_* メソッドをディスパッチする
    同時に、簡易的なシンボルテーブルを持ち、
      - 変数名 -> 型
    のマッピングを扱う。
    """

    def __init__(self):
        # 変数名 -> 型名 (str) の簡易マッピング
        self.symbol_table = {}

    def set_symbol_type(self, name: str, t: str):
        self.symbol_table[name] = t

    def get_symbol_type(self, name: str) -> str:
        return self.symbol_table.get(name, "i32")  # デフォルトをi32にしておく

    def visit(self, node: ast.AST) -> Any:
        method = f"visit_{node.__class__.__name__}"
        visitor = getattr(self, method, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node: ast.AST) -> None:
        """
        各ノードに対応する visit_* が未定義の場合、ここにフォールバック
        未実装の構文要素があればエラーを出す
        """
        raise NotImplementedError(f"Node type {type(node).__name__} not implemented")
