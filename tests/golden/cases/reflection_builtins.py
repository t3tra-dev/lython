# What this pins: `hasattr`, `getattr`, `setattr` and `callable` -- four names
# that were simply unbound, and four questions this compiler can answer without
# asking the runtime anything.
#
#     print(hasattr(x, "v"))     # unresolved name 'hasattr'
#     print(getattr(x, "v"))     # unresolved name 'getattr'
#     setattr(x, "v", 2)         # unresolved name 'setattr'
#     print(callable(f))         # unresolved name 'callable'
#
# The attribute either exists on the static class or it does not, and a value
# either has a callable contract or it does not, so both fold. `getattr` with a
# literal name IS the attribute lookup written as a call, so it is rewritten to
# one and every rule about attributes applies to it unchanged; `setattr` is the
# same rewrite on the store side, so the field's declared type and the release
# of the value it replaces come from the assignment path, not from a second
# implementation of it.
#
# Why this must run: the folds decide printed truth values, and the argument
# still has to be EVALUATED -- `hasattr(make(), "v")` calls make(), once, which
# only a counter shows. `setattr` has to be read back to prove the store landed
# on the field the name spells, and it answers None like CPython's does.
#
# ⛔ A SUBCLASS CAN ONLY ADD, which makes the two answers asymmetric. A True
# stands: the base has the attribute, so every instance does. A False is refused
# when the class has a subclass, because the subclass may define exactly that
# name -- `hasattr(v, "go")` where `v: A = B()` and B defines `go` is True in
# CPython, and answering False would be the silent wrong answer this project
# exists to avoid. The refusal names the subclass to look at.
#
# ⛔ A computed attribute name is refused for the same reason: there is nothing
# static to answer -- for `setattr` too, where the store would otherwise have no
# field to land on. So is `getattr(x, "v", default)`, whose arm choice would
# need the hasattr fold.


class C:
    def __init__(self, v: int) -> None:
        self.v = v
        self.tag = "a"

    def go(self) -> int:
        return self.v + 1


def plain() -> int:
    return 1


calls = 0


def make() -> C:
    global calls
    calls += 1
    return C(2)


x = C(1)
print(hasattr(x, "v"), hasattr(x, "go"), hasattr(x, "nope"))
print(hasattr("s", "upper"), hasattr([1], "append"), hasattr(1, "bit_length"))
print(getattr(x, "v"), getattr(x, "go")())
print(callable(plain), callable(C), callable(x), callable(1), callable("s"))
print(hasattr(make(), "v"), calls)
print(setattr(x, "v", 7), x.v, getattr(x, "v"), x.go())
field = "tag"
i = 0
while i < 3:
    setattr(x, field, "b" * i)
    i += 1
print(x.tag, len(x.tag))
