# WHAT: `fmt % t` with `t` a tuple held in a name -- a pair, a one-element
# tuple, a tuple parameter, the items of a dict -- formats the tuple's items,
# as a tuple written in place does; an `object` that holds a tuple is spread
# too and one that does not is the single argument; a count that does not
# match raises CPython's TypeError at run time.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: for an `object` and for a
# one-element spelling the count is the run time's. A tuple in a name was
# formatted as one argument: `"[%s]" % ("only",)` held in a name printed
# `[('only',)]` where CPython prints `[only]`, silently.
kv = ("a", 1)
print("%s=%r" % kv)
one = ("only",)
print("[%s]" % one)


def f(pair: tuple[str, int]) -> str:
    return "%s:%d" % pair


def g(t: tuple[str]) -> str:
    return "<%s>" % t


print(f(("x", 2)), g(("y",)))
pairs = {"k": 1, "j": 2}
print(["%s=%r" % item for item in sorted(pairs.items())])


def show(o: object) -> str:
    try:
        return "[%s]" % o
    except TypeError as e:
        return "TypeError: " + str(e)


print(show(5), show("s"), show((7,)), show(("a", "b")), show(()))
