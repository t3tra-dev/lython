# WHAT: the container of a slice assignment or deletion is evaluated once, as
# CPython evaluates `f()[a:b] = v` -- the call that produces the list runs one
# time whether the slice is written out or is a slice object.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: the defect was a second
# evaluation of the target expression, which shows only as a side effect the
# program counts: `get(ys)[0:1] = [9]` called get twice and printed 2.

calls: list[str] = []


def get(xs: list[int], why: str) -> list[int]:
    calls.append(why)
    return xs


ys = [1, 2, 3, 4]
get(ys, "assign")[0:1] = [9]
print(calls, ys)
del get(ys, "delete")[1:2]
print(calls, ys)
get(ys, "object")[slice(0, 1)] = [5, 6]
print(calls, ys)
del get(ys, "object delete")[slice(None, None, 2)]
print(calls, ys)
