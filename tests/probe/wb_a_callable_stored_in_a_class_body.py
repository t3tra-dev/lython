# A class attribute holding a FUNCTION -- a lambda or a def by name -- cannot be
# read at all:
#
#     class C:
#         V: Callable[[int], int] = lambda n: n + 1
#     print(C.V(1))
#     # static type !py.type<!py.contract<"C">> does not provide manifest
#     # method 'V'
#
# MEASURED (2026-09-06, RelWithDebInfo, today's tree) by sweeping every
# annotation kind through one shape -- declare it in a class body, read it from
# a method AND from module scope. Sixteen kinds; this is the only one:
#
#   int / str / float / bool / bytes ......... correct
#   list / dict / set / tuple ................ correct
#   a class instance, a list of them ......... correct
#   Optional[int] / None / type[Box] ......... correct
#   through `self.V` instead of `C.V` ........ correct
#   Callable[[int], int] ..................... this file, both spellings
#
# ⭐ THE STORABILITY RULE IS NOT WHERE IT IS. Adding `CallableType` to the
# slot test in `emitClassContract` moves nothing -- the message is unchanged at
# module scope and inside a method -- so the attribute never reaches that walk.
# The class-body collection that fills `staticAttrNames` is where to look.
#
# ⛔ AND IT IS A SHAPE QUESTION BEFORE IT IS A STORAGE ONE. CPython stores a
# plain function in a class dict as an UNBOUND method: `C.V(1)` is the call
# written here, and `C().V(1)` passes the instance as the first argument --
# which is exactly what `@staticmethod` exists to opt out of. A slot that hands
# back a function object answers the first and gets the second wrong, so the
# repair has to decide which descriptor the attribute is before it decides
# where it lives.
#
# The module-global twin of this shape was a missing arm and nothing more
# ([[lython-module-storage]]): `DOUBLE: Callable[[int], int] = lambda ...` at
# module scope got its cell on the same day this stayed refused.
from typing import Callable


class C:
    V: Callable[[int], int] = lambda n: n + 1

    def use(self) -> int:
        return C.V(1)


print(C.V(1), C().use())
