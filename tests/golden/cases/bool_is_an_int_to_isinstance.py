# What this pins: the two hierarchy predicates over the bool/int pair, and that
# narrowing through them still yields a usable value.
#
#     print(issubclass(bool, int))   # printed False; CPython prints True
#     print(isinstance(True, int))   # printed False; CPython prints True
#
# Both asked assignability, and assignability is deliberately narrower: a bool is
# one truth bit while an int is a three-value bundle, so a bool VALUE cannot be
# stored where an int is expected without a conversion. That is an ABI fact.
# `issubclass` and `isinstance` ask about the CLASS hierarchy, where bool's base
# IS int -- so the answer came back False, with no diagnostic, for the one pair
# where CPython says True.
#
# Why this must run: the answer is a printed truth value, and the pair that was
# wrong is answered by a compile-time fold -- there is nothing to read in the IR
# that says which way it folded. `widen` is here for the other half: after
# `isinstance(b, int)` narrows, the arm still has to produce 2 and not a
# reinterpreted truth bit.
#
# ⛔ The numeric tower is NOT the rule. `issubclass(int, float)` stays False,
# which is CPython's answer too even though the tower converts one to the other,
# and `isinstance(1, bool)` stays False: an int value's class is int. One rung,
# in the direction the hierarchy actually has it.


def widen(b: bool) -> int:
    if isinstance(b, int):
        return b + 1
    return 0


def which(x: bool | str) -> str:
    if isinstance(x, int):
        return "int-ish"
    return "str"


print(issubclass(bool, int), issubclass(int, bool), issubclass(int, float))
print(isinstance(True, int), isinstance(1, bool), isinstance(True, bool))
print(widen(True), widen(False))
print(which(True), which("a"))
