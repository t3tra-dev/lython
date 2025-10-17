import ast

from ..mlir import ir
from ._base import BaseVisitor


class ArgumentsVisitor(BaseVisitor):
    """
    ```asdl
    arguments = (arg* posonlyargs, arg* args, arg? vararg, arg* kwonlyargs,
                 expr* kw_defaults, arg? kwarg, expr* defaults)
    ```
    """

    def __init__(self, ctx: ir.Context) -> None:
        super().__init__(ctx)

    def visit_arguments(self, node: ast.arguments) -> None:
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
