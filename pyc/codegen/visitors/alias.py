from .base import BaseVisitor


class AliasVisitor(BaseVisitor):
    """
    ```asdl
    -- import name with optional 'as' alias.
    alias = (identifier name, identifier? asname)
             attributes (int lineno, int col_offset, int? end_lineno, int? end_col_offset)
    """
    pass
