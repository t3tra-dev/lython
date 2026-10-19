# DEVIATION, deliberate and recorded. A call through a class value reaches a
# class whose `__init__` declares a parameter of another type than the
# argument the call passes. CPython builds the object anyway -- annotations
# are inert there -- and Lython raises a TypeError saying why it cannot.
#
# ⛔ WHY NOT CPython'S ANSWER. The object would hold a str in a field declared
# list[int], and a field here has the layout of its declared type; there is
# nothing to build. The arm is compiled at all because the program is
# otherwise well typed: `cls(msg)` over `type[BaseException]` names every
# exception class of the program, and refusing the call for one unrelated
# class with its own `__init__` -- what stood here -- refused unittest's
# `self.failureException(msg)` in every test file that declares such a class.
# CPython reaches this arm only if that class is the one called, and so does
# Lython: then it raises.
#
# ⛔ AND ONLY THE TYPE. An argument COUNT the `__init__` cannot take raises
# CPython's own TypeError, word for word
# (cases/a_class_value_call_raises_where_the_class_cannot_take_it).
#
# Measured 2026-10-20: CPython prints Carrier(), this prints the TypeError.
# differential: expect-wrong an argument of a type the __init__ does not declare raises TypeError
class Carrier(Exception):
    def __init__(self, payload: list[int]) -> None:
        super().__init__()
        self.payload = payload


def build(cls: type[BaseException], msg: str) -> BaseException:
    return cls(msg)


print(repr(build(ValueError, "v")))
print(Carrier([1]).payload)
try:
    print(repr(build(Carrier, "c")))
except TypeError as e:
    print("TypeError:", e)
