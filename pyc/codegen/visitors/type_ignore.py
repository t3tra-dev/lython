import ast

from pyc.codegen.ir.builder import IRBuilder
from .base import BaseVisitor


class TypeIgnoreVisitor(BaseVisitor):
    """
    ```asdl
    type_ignore = TypeIgnore(int lineno, string tag)
    ```
    """
    def __init__(self, builder: IRBuilder):
        super().__init__(builder)

    def visit_type_ignore(self, node: ast.type_ignore) -> None:
        """
        ```asdl
        type_ignore = TypeIgnore(int lineno, string tag)
        ```
        """
        raise NotImplementedError("type_ignore not implemented")
