# What this pins: `==` and `!=` where the two sides cannot meet.
#
# Two shapes, one answer. A comparison across two value families --
# `"a" == 1` -- went looking for a runtime method with the other side's shape
# and did not find it:
#
#     print("a" == 1)
#     # cannot adapt builtins.int to runtime input 2 of builtins.str.__eq__
#
# The manifest is right to have only `str.__eq__(str, str)`; there is no
# runtime question here. CPython answers it with the NotImplemented rule --
# neither side's __eq__ accepts the other, so it falls back to identity and is
# False -- and both types are known where the comparison is emitted.
#
# And a UNION compares per member, decided by the tag. Only Optional[T] with
# exactly one present member was answered; `rec["age"] == 30` on a record
# literal was "static type !py.union<int, str> does not provide manifest method
# '__eq__'". Under the tag each member is concrete, and the arms whose member
# is a different family from the other operand fold to the constant above --
# which is why the two belong in one case: the union dispatch does not work
# without the fold.
#
# Why this needs to run rather than assert on a diagnostic: a fold is a VALUE,
# and the wrong constant compiles just as well as the right one. `1 == True` is
# True and `1 == "a"` is False, and both are answered without a call. Under the
# tag, choosing the wrong arm prints True where CPython prints False.
#
# ⛔ Only families whose cross-family answer is unconditionally False are
# folded. Set-vs-frozenset is the counterexample that keeps container kinds
# out: `{1} == frozenset({1})` is True in CPython, and from the fold's vantage
# a set and a frozenset look exactly as different as a str and an int. It is
# still refused for its own reason ("cannot adapt builtins.frozenset to runtime
# input 1 of builtins.set.__eq__"), recorded in
# tests/probe/wb_grid_leftovers_2026_08_16.py.
#
# Every expected line is python3.14's.

# --- across families, both directions and both operators -------------------
print("a" == 1, 1 == "a", "a" != 1, 1 != "a")
print(b"a" == "a", "a" == b"a", b"a" != "a")
print(None == 1, 1 == None, None != "a")

# --- WITHIN a family, which is where the numeric tower must still answer ---
print(1 == 1.0, 1.0 == 1, True == 1, 1 == True, False == 0)
print(2.5 == 2.5, "a" == "a", b"a" == b"a", None == None)


# --- a record literal, which is the union shape ---------------------------
rec = {"name": "ann", "age": 30}
print(rec["name"] == "ann", rec["name"] == 30)
print(rec["age"] == 30, rec["age"] == "ann")
print(rec["name"] != "ann", rec["age"] != 30)


# --- three members --------------------------------------------------------
xs = [1, "a", 2.5]
print(xs[0] == 1, xs[1] == "a", xs[2] == 2.5)
print(xs[0] == "a", xs[1] == 1, xs[2] == 1)


# --- a union on BOTH sides ------------------------------------------------
def pick(f: int) -> int | str:
    if f == 0:
        return 1
    return "a"


a = pick(0)
b = pick(1)
print(a == b, a == 1, b == "a", a != b)
print(pick(0) == pick(0), pick(1) == pick(1))


# --- Optional[T], which used to be the only union answered -----------------
from typing import Optional


def same(x: Optional[int], y: Optional[int]) -> bool:
    return x == y


print(same(1, 1), same(1, 2), same(None, None), same(1, None), same(None, 1))

vals: list[int | None] = [1, None]
print(vals[0] == 1, vals[1] == 1, vals[0] == None, vals[1] == None)
print(vals[0] is None, vals[1] is None)


# --- THE CONTROL: a hand-written __eq__ still answers for itself -----------
# It accepts object and may say anything, so nothing here may be folded.
class P:
    def __init__(self, v: int) -> None:
        self.v = v

    def __eq__(self, o: object) -> bool:
        return True


print(P(1) == P(2), P(1) == 5, P(1) == "x")
