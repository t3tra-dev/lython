from .base import BaseVisitor


class ArgVisitor(BaseVisitor):
    """
    ```asdl
    arg = (identifier arg, expr? annotation, string? type_comment)
           attributes (int lineno, int col_offset, int? end_lineno, int? end_col_offset)
    """
    pass
