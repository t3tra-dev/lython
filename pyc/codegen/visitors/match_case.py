from .base import BaseVisitor


class MatchCaseVisitor(BaseVisitor):
    """
    ```asdl
    match_case = (pattern pattern, expr? guard, stmt* body)
    """
    pass
