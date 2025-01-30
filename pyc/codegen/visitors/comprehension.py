from .base import BaseVisitor


class ComprehensionVisitor(BaseVisitor):
    """
    ```asdl
    comprehension = (expr target, expr iter, expr* ifs, int is_async)
    """
    pass
