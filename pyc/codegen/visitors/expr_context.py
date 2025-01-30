from .base import BaseVisitor


class ExprContextVisitor(BaseVisitor):
    """
    ```asdl
    expr_context = Load | Store | Del
    """
    pass
