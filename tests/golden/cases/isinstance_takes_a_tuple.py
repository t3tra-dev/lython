# What this pins: `isinstance(x, (A, B))` -- CPython's own spelling for "any of
# these" -- and the narrowing that follows it in both directions.
#
#     if isinstance(v, (int, float)):
#     # second argument to isinstance must be a statically resolved class type
#
# The target was read as ONE class and a tuple is not one, so the whole form was
# refused. Each element still has to be a statically resolved class; what was
# missing is looking inside.
#
# Why this must run: the answer is a truth value AND a narrowed type, and the
# narrowing is what the arithmetic in each arm depends on. `kind` needs the true
# arm to hold `int | float` (both members answer), `tail` needs the FALSE arm to
# hold the one member no element selected -- `x + 0.5` is a float's `__add__`,
# and it only resolves if the else-arm type is float alone.
#
# ⛔ A tuple element that would need a runtime class test is refused rather than
# merged: the tests are per-member ops over one union value, and a class test is
# not one of them. Splitting the tuple into separate isinstance calls still
# works, which is what the refusal says to do.


def kind(v: int | str | float) -> str:
    if isinstance(v, (int, float)):
        return "number"
    return "text"


def tail(x: int | str | float) -> float:
    if isinstance(x, (int, str)):
        return 0.0
    return x + 0.5


v: int | str = 3
if isinstance(v, (str,)):
    print("str")
else:
    print("int", v + 1)

print(kind(1), kind(2.5), kind("a"))
print(tail(2.5), tail(1))
print(isinstance(v, (int, str)), isinstance(v, (bytes, float)))
