# WHAT: a class (`int`, `float`, a program class) passed where
# `Callable[[str], object] | None` is declared -- argparse's `type=int` --
# called through the parameter, and read back as None when not given.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: the value the parameter
# holds is a forwarder made for the class, and what calling it gives is a
# run-time fact.
#
# ⛔ A class met `Callable[...]` but not `Callable[...] | None`: the union
# rule asked each member, and no member rule took a class.
from collections.abc import Callable


class Celsius:
    def __init__(self, text: str) -> None:
        self.degrees = float(text)


def convert(text: str, kind: Callable[[str], object] | None = None) -> str:
    if kind is None:
        return "raw " + text
    value = kind(text)
    if isinstance(value, Celsius):
        return "celsius " + str(value.degrees)
    return repr(value)


print(convert("12", int))
print(convert("1.5", float))
print(convert("21", Celsius))
print(convert("12"))
