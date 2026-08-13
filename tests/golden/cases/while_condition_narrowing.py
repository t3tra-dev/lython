# Two repairs, and the second needed the first.
#
# (1) A UNION carried local is released once per iteration by the loop's own
# edge bookkeeping, and the matching acquisition on the way in is placed by
# the ownership pass -- for every type except a union, whose release is
# guarded by its tag and which that pass skips. With neither half the first
# back edge freed a value the loop never owned: `while s is not None: s =
# None` over a str|None PARAMETER freed the caller's string and printed an
# empty line, or aborted with "Ly_DecRef observed non-positive refcount".
#
# (2) On top of that, the body now sees what the condition proves. `while n
# is not None:` left n a union inside its own body, so `total += n` was
# refused for an operand the loop only enters with when it is an int -- while
# `if n is not None:` and the conditional expression both narrowed it.
def ints(n: int | None) -> int:
    total = 0
    while n is not None:
        total += n
        n = None
    return total


def strs(s: str | None) -> str:
    out = ""
    while s is not None:
        out = out + s
        s = None
    return out


def lists(xs: list[int] | None) -> int:
    total = 0
    while xs is not None:
        for v in xs:
            total += v
        xs = None
    return total


# The borrowed-parameter shape from (1): the loop must not free the caller's
# value, and the string built before the loop must survive it.
def keeps_caller_value(s: str | None) -> str:
    out = "start"
    while s is not None:
        s = None
    return out


# Rebinding the narrowed name to another union value: the header re-tests it,
# so the second value is narrowed on the next iteration too.
def pick(v: int | None, other: int | None) -> int:
    seen = 0
    while v is not None:
        seen += v
        v = other
        other = None
    return seen


# isinstance over a union narrows the same way.
def isinst(v: int | str) -> int:
    seen = 0
    while isinstance(v, int):
        seen += v
        break
    return seen


print(ints(5), ints(None))
print(strs("ab"), strs(None) == "")
print(lists([1, 2, 3]), lists(None))
print(keeps_caller_value("ab"), keeps_caller_value(None))
print(pick(1, 2), pick(None, 3))
print(isinst(7), isinst("x"))
