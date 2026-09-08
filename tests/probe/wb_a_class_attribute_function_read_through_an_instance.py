# OPEN, and it is the descriptor corner rather than a storage one. A class
# attribute holding a function is now readable and callable through the CLASS
# (cases/a_class_attribute_that_holds_a_function); through an INSTANCE it is
# refused:
#
#     class C:
#         def __init__(self) -> None: self.k = 10
#         V: "Callable[[C, int], int]" = lambda self, n: self.k + n
#     C().V(1)      # CPython 11; here: "'V' is a class attribute holding a
#                   # function, and reading it through an instance binds the
#                   # receiver, which is not supported"
#
# MEASURED 2026-09-09:
#
#   C.V(1) / Sub.V(1) / type(c).V(1) / a bound `t = type(c)` .... correct
#   f = C.V; f(1) ................................................ correct
#   Bound.SCALE(obj, n) (the unbound spelling written out) ....... correct
#   c.V(1) ....................................................... refused
#   the same shape as @staticmethod ............................... correct
#
# ⭐ WHAT IS MISSING IS A DESCRIPTOR, not a slot. CPython's function objects
# implement `__get__`, so `c.V` is a NEW callable with the instance already
# bound -- one more value shape (a bound function object over a class-attribute
# function) plus the arity rebate at the call. The emitter has a bound-method
# object for real methods (`emitMethodObject`); what it has no shape for is
# binding a receiver to a value that arrived as a class ATTRIBUTE.
#
# ⛔ NOT the same as `self.fn(...)` for a FIELD holding a callable, which works
# and must keep working: a field's value is not a descriptor, and CPython does
# not bind a receiver to it either.
from typing import Callable


class C:
    def __init__(self) -> None:
        self.k: int = 10

    V: "Callable[[C, int], int]" = lambda self, n: self.k + n


print(C().V(1))
