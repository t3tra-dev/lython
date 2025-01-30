from .base import BaseVisitor


class OperatorVisitor(BaseVisitor):
    """
    ```asdl
    operator = Add | Sub | Mult | MatMult | Div | Mod | Pow | LShift
                 | RShift | BitOr | BitXor | BitAnd | FloorDiv
    """
    pass
