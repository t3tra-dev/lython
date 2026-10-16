# What: a function imported under the spelling of a builtin CLASS -- `range`,
# `str`, `int` -- is what a call of that name reaches. The five plain builtins
# (a_builtin_shadowed_by_an_import) were already shadowed; a class spelling was
# resolved to the class first, in the call, in the name read and in the
# inference, so `for v in range(7)` silently counted 0..6 where CPython prints
# 7 and 8, and `str(3)` / `int("a")` were refused as constructor calls.
#
# Only running it can catch this: the range loop compiles either way and the
# answer is the difference.
from a_module_of_shadowing_class_spellings import int, range, str


def returned() -> list[int]:
    return range(1)


class Holder:
    def tens(self, n: int) -> int:
        return str(n)


for v in range(7):
    print("loop", v)
bound = range(3)
print("bound", bound)
print("comprehension", [w for w in range(5)])
print("returned", returned())
print("method", Holder().tens(4))
print("str", str(3), "int", int("a"))
as_value = range
print("as a value", as_value(9))
