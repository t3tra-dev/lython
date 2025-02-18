import ast

from .base import BaseVisitor
from ...codegen.ir.builder import IRBuilder


class TypeParamVisitor(BaseVisitor):
    """
    ```asdl
    type_param = TypeVar(identifier name, expr? bound, expr? default_value)
    ```
    """
    def __init__(self, builder: IRBuilder):
        super().__init__(builder)

    def visit_tyoe_param(self, node: ast.type_param) -> None:
        """
        ```asdl
        type_param = TypeVar(identifier name, expr? bound, expr? default_value)
        ```
        """
        raise NotImplementedError("type_param not implemented")
