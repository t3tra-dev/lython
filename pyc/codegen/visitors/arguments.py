from .base import BaseVisitor


class ArgumentsVisitor(BaseVisitor):
    """
    ```asdl
    arguments = (arg* posonlyargs, arg* args, arg? vararg, arg* kwonlyargs,
                 expr* kw_defaults, arg? kwarg, expr* defaults)
    """
    pass
