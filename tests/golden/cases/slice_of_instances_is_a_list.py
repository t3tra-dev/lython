# What this pins: a slice of a list of instances, used where its TYPE decides.
#
#     class V:
#         def __repr__(self) -> str:
#             return "v"
#     vs = [V(), V()]
#     print(vs[0:2])        # printed v; CPython prints [v, v]
#
# Byte-identical to what `print(vs[0])` prints, because the print/repr dispatch
# asked the lenient inference walk for the operand's type, got `V` -- the ELEMENT
# -- found `V.__repr__` and inlined it with the slice's list handle as the
# receiver. `repr(...)` and `str(...)` were wrong the same way, `vs[0:0]` printed
# an element where CPython prints `[]`, and the element's `__repr__` ran exactly
# once for a three-element slice. `list(vs[0:2])` failed to compile for the same
# reason, and `vs[0:2].k` reached the lowering as a field read.
#
# The strict walk has been right since the generator yield-type fix ("A SLICE IS
# `__getslice__`, NOT `__getitem__`"), and the lenient one was deliberately left
# alone there because correcting it broke `a[bump():3] += [99]`. That was the
# augmented-assignment route wanting not a different TYPE but not to run: a slice
# target's `+=` is a slice ASSIGNMENT in CPython, and reading `a[i:j]` produces a
# new list, so rewriting it to `a[i:j].extend([99])` extends a copy and the splice
# disappears. With the in-place route declining slice targets, the lenient walk
# answers correctly for everyone.
#
# Why this needs to run rather than assert on a diagnostic: the failure printed a
# plausible value at exit 0. The element counter below is the assertion that
# matters -- it says how many times the element's `__repr__` ran, which is what
# separates "formats the list" from "formats one element".
#
# ⛔ `list[int]` slices always printed correctly, and that is why this went
# unnoticed: an int element type reaches no source-class method, so the wrong
# answer never selects anything. It takes a user class to become visible.
#
# Every expected line is python3.14's.


class V:
    def __init__(self, n: int) -> None:
        self.n = n

    def __repr__(self) -> str:
        return "V" + str(self.n)


vs = [V(1), V(2), V(3)]

# --- the operand positions that were wrong ---------------------------------
print(vs[0:2])
print(vs[1:3])
print(vs[0:0])
print(vs[:])
print(vs[::2])
print(repr(vs[0:2]), str(vs[0:2]))
print(list(vs[0:2]))
print(f"{vs[0:2]}")


# --- how many times the element repr runs ---------------------------------
calls = 0


class Counted:
    def __repr__(self) -> str:
        global calls
        calls += 1
        return "c"


cs = [Counted(), Counted(), Counted()]
print(cs[0:3])
print(calls)
print(cs[0:1])
print(calls)


# --- through a function and a method, and with runtime bounds -------------
def show(items: list[V], lo: int, hi: int) -> None:
    print(items[lo:hi])


show(vs, 0, 2)
show(vs, 1, 3)


class Holder:
    def __init__(self, items: list[V]) -> None:
        self.items = items

    def head(self, n: int) -> None:
        print(self.items[0:n])


Holder(vs).head(2)


# --- the slice really is a list afterwards --------------------------------
part = vs[0:2]
print(len(part), part[0], part + [V(9)], sorted([1, 2]))
print(len(vs[0:2]), vs[0:2][1])


# --- THE CONTROL: int elements, which always worked ----------------------
xs = [1, 2, 3]
print(xs[0:2], repr(xs[0:2]), len(xs[0:2]))


# --- and the augmented slice assignment the lenient answer used to serve --
a = [1, 2, 3, 4, 5]
a[1:3] += [99]
print(a)
b = [1, 2, 3]
b[0:0] += [0]
print(b)
