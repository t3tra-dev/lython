# What this pins: a method's union parameter called with one member.
#
#     class Box:
#         def take(self, n: int | None) -> int:
#             if n is None:
#                 return -1
#             return n
#     Box().take(None)
#     # cannot adapt runtime bundle types.NoneType with physical values ...
#
# An inlined body binds the argument VALUE, so the parameter held a
# `literal<None>` and the body's `n is None` narrowing had nothing to narrow --
# the union it was written against never existed at that call. The free-function
# spelling works because its call emits operands against the declared callable,
# which wraps the member; a DEFAULT of None works for the same reason. So the
# inlined path wraps too, whenever the declared parameter is a union and the
# argument is not.
#
# `collections.Counter.most_common(None)` is this defect reached through the
# shipped stdlib: its parameter is `int | None` and the explicit None is how
# CPython's own signature is exercised.
#
# Why this needs to run rather than assert on a diagnostic: the wrap decides which
# ARM the body takes. Wrapping into the wrong member, or narrowing to the wrong
# one, compiles and returns the other branch's value -- so every section calls the
# same method with each member and prints both answers.
#
# Every expected line is python3.14's.

import collections


class Box:
    def take(self, n: int | None) -> int:
        if n is None:
            return -1
        return n

    def label(self, v: int | str) -> str:
        if isinstance(v, int):
            return "i" + str(v)
        return "s" + v

    def kw(self, *, n: int | None = None) -> int:
        if n is None:
            return -1
        return n


b = Box()

# --- the literal member, each arm ------------------------------------------
print(b.take(None), b.take(5))
print(b.label(5), b.label("x"))

# --- a union-typed variable, which always worked ---------------------------
v: int | None = 7
w: int | None = None
print(b.take(v), b.take(w))

# --- keyword-only, with and without the default ---------------------------
print(b.kw(), b.kw(n=5), b.kw(n=None))

# --- through the shipped stdlib -------------------------------------------
c: collections.Counter = collections.Counter(["a", "b", "a"])
print(c.most_common(None))
print(c.most_common(1))


# --- THE CONTROL: the free function, which was always right ---------------
def free_take(n: int | None) -> int:
    if n is None:
        return -1
    return n


print(free_take(None), free_take(5))
