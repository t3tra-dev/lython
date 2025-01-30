from .base import BaseVisitor


class UnaryOpVisitor(BaseVisitor):
    """
    ```asdl
    unaryop = Invert | Not | UAdd | USub
    """
    pass
