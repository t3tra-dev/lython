# What this pins: a `try` whose `else` clause contains an `if`, a `while` or a
# `for`.
#
#     try:
#         x = 1
#     except ValueError:
#         pass
#     else:
#         if x > 0:
#             print("a")
#     # empty block: expect at least a terminator
#
# The walk branched the else clause to the continuation by terminating the
# block the clause STARTED in. A nested statement splits that block, so by the
# time the clause is over the builder is somewhere else -- the starting block
# already had its terminator from the split, and the block actually holding the
# insertion point got none. A straight-line else body always worked, which is
# why it stood: with no split the two blocks are the same one.
#
# Why this needs to run rather than assert on a diagnostic: what the repair
# decides is where the else clause's LAST block goes, and a wrong answer is a
# skipped statement rather than a crash. Each shape below prints from inside
# the nested statement AND after the try, so a clause that fell through to the
# wrong continuation loses a line.
#
# ⛔ Every name the else clause reads is bound BEFORE the try. A binding
# CREATED inside a try body does not escape it here -- `n = v` in the body then
# `n` in the else is "unresolved name 'n'" -- which is a separate defect from
# this one and is recorded in tests/probe/wb_grid_leftovers_2026_08_16.py.
# Rebinding a pre-existing local works, which is what these do.
#
# Every expected line is python3.14's.


# --- an if, with and without its own else ---------------------------------
def plain_if(v: int) -> str:
    out = "-"
    n = 0
    try:
        n = v
    except ValueError:
        return "raised"
    else:
        if n > 0:
            out = "positive"
    return out


def if_else(v: int) -> str:
    out = "-"
    n = 0
    try:
        n = v
    except ValueError:
        return "raised"
    else:
        if n > 0:
            out = "positive"
        else:
            out = "other"
    return out


print(plain_if(1), plain_if(-1))
print(if_else(1), if_else(-1))


# --- a while and a for ------------------------------------------------------
def counted(k: int) -> int:
    total = 0
    n = 0
    try:
        n = k
    except ValueError:
        return -1
    else:
        i = 0
        while i < n:
            total += i
            i += 1
    return total


def over_list(xs: list[int]) -> int:
    total = 0
    ys: list[int] = []
    try:
        ys = xs
    except ValueError:
        return -1
    else:
        for v in ys:
            total += v
    return total


print(counted(4), counted(0))
print(over_list([1, 2, 3]), over_list([]))


# --- the else runs only when the body did not raise ------------------------
def guarded(k: int) -> str:
    log: list[str] = []
    try:
        if k == 0:
            raise ValueError("zero")
        log.append("body")
    except ValueError:
        log.append("handler")
    else:
        if len(log) > 0:
            log.append("else")
    return ",".join(log)


print(guarded(1))
print(guarded(0))


# --- nested two deep, and a finally alongside ------------------------------
def deep(k: int) -> str:
    marks: list[str] = []
    n = 0
    try:
        n = k
    except ValueError:
        marks.append("handler")
    else:
        if n > 0:
            if n > 1:
                marks.append("big")
            else:
                marks.append("one")
    finally:
        marks.append("finally")
    return ",".join(marks)


print(deep(2))
print(deep(1))
print(deep(0))


# --- THE CONTROL: a straight-line else, which always worked ----------------
try:
    value = 5
except ValueError:
    print("raised")
else:
    print("straight", value)
print("after")
