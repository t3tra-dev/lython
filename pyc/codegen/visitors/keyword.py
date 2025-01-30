from .base import BaseVisitor


class KeywordVisitor(BaseVisitor):
    """
    ```asdl
    -- keyword arguments supplied to call (NULL identifier for **kwargs)
    keyword = (identifier? arg, expr value)
               attributes (int lineno, int col_offset, int? end_lineno, int? end_col_offset)
    """
    pass
