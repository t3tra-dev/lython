from .base import BaseVisitor


class CmpOpVisitor(BaseVisitor):
    """
    ```asdl
    cmpop = Eq | NotEq | Lt | LtE | Gt | GtE | Is | IsNot | In | NotIn
    """
    pass
