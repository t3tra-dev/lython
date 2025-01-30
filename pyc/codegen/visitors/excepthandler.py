from .base import BaseVisitor


class ExceptHandlerVisitor(BaseVisitor):
    """
    ```asdl
    excepthandler = ExceptHandler(expr? type, identifier? name, stmt* body)
                    attributes (int lineno, int col_offset, int? end_lineno, int? end_col_offset)
    """
    pass
