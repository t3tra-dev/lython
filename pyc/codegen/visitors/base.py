import ast
from typing import Any

__all__ = ["BaseVisitor"]


class BaseVisitor:
    """
    ベースとなるVisitorクラス
    すべてのノードタイプに対して visit_* メソッドをディスパッチする
    """

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
