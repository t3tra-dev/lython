import ast

from .base import BaseVisitor
from ...codegen.ir.builder import IRBuilder


class ArgumentsVisitor(BaseVisitor):
    """
    ```asdl
    arguments = (arg* posonlyargs, arg* args, arg? vararg, arg* kwonlyargs,
                 expr* kw_defaults, arg? kwarg, expr* defaults)
    ```
    """
    def __init__(self, builder: IRBuilder):
        super().__init__(builder)

    def visit_argumentes(self, node: ast.arguments) -> None:
        """
        ```asdl
        arguments = (
            arg* posonlyargs,
            arg* args,
            arg? vararg,
            arg* kwonlyargs,
            expr* kw_defaults,
            arg? kwarg,
            expr* defaults
        )
        ```
        """
        raise NotImplementedError("arguments not implemented")
