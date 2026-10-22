# WHAT: a class that defines `__getattribute__` answers every read on its
# instances, including a field it declares and set in `__init__`.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: the answer is the method's
# value, computed when the read runs.
#
# ⛔ A field the class declares was read directly, past the method, so
# `Fixed().x` printed 1 where CPython prints 42 -- silently.


class Fixed:
    def __init__(self) -> None:
        self.x = 1

    def __getattribute__(self, name: str) -> int:
        return 42


class Echo:
    def __init__(self) -> None:
        self.field = "stored"

    def __getattribute__(self, name: str) -> str:
        return "<" + name + ">"


print(Fixed().x)
e = Echo()
print(e.field, e.anything)
