# What this pins: None where the static type is `object` -- a list[object]
# slot, a dict value, an `object` parameter.
#
#     xs: list[object] = [1, None]
#     print(xs[1])            # Ly_DecRef observed non-positive refcount
#     def f(v: object) -> int: return 1
#     print(f(None))          # the same abort, with the body never reading v
#
# None's payload handle is sixteen zero words -- no class, no entity, and the
# owned flag saying the slot owns nothing -- and TWO of those zeros were read as
# something else. Word 0 is a refcount when the handle is a standalone box
# rather than a container slot, so LyObject_DecRef read the zero as "already
# dead"; and LyObject_FromSlot stamped the owned flag to 1 on the way out of a
# slot whose entity the retain beside it had skipped, so the box owed a release
# it never took.
#
# Why this must run: both are refcount arithmetic. Nothing is visibly wrong in
# the IR -- the abort is the runtime catching the second decrement -- and the
# loop at the end runs the read often enough that a leak or an over-release
# shows up either as a crash or in tests/leak_gate.py, which reads 0 here.
#
# ⛔ `x is None` ON AN ERASED VALUE IS A RUNTIME TEST, not the compile-time fold
# every other type gets. None is a singleton, so a concrete type is never it and
# the fold is right; `object` is not a type but the absence of one, and the fold
# answered False for a slot that held None. It became visible only once the two
# zeros above were fixed -- before that the program aborted before printing the
# wrong answer.
def which(v: object) -> str:
    if v is None:
        return "none"
    if v == 1:
        return "one"
    return "other"


def untouched(v: object) -> int:
    return 1


xs: list[object] = [1, None, "a", 2.5, True]
print(xs[0], xs[1], xs[2], xs[3], xs[4])
print(xs[0] is None, xs[1] is None, xs[2] is None)
print(xs[0] is not None, xs[1] is not None)
print(xs[1] == 1, xs[1] == "a", 1 == xs[1])
print(which(1), which(None), which("z"))
print(untouched(None), untouched(1))
print(str(xs[1]), len(str(xs[1])))

d: dict[str, object] = {"a": None, "b": 2}
print(d["a"], d["a"] is None, d["b"] is None)

nones = 0
i = 0
while i < 200:
    if xs[1] is None:
        nones += 1
    if xs[0] is not None:
        nones += 1
    nones += len(str(xs[1]))
    i += 1
print("nones", nones)
