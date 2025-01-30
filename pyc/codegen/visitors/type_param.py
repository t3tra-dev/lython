from .base import BaseVisitor


class TypeParamVisitor(BaseVisitor):
    """
    ```asdl
    type_param = TypeVar(identifier name, expr? bound, expr? default_value)
    """
    pass
