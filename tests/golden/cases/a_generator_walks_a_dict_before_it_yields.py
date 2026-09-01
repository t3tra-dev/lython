# What: the loop's iterator is a compile-time token whose position lives in a
# function-level cell, and a generator threads the loop's values through the
# resume function's block arguments -- so only running it shows the token
# reached `__next__` with its cell rather than as a bare protocol value.
def keys(d: "dict[str, int]"):
    ks: "list[str]" = []
    for k in d:
        ks.append(k)
    for k2 in ks:
        yield k2


print(list(keys({"a": 1, "b": 2, "c": 3})))


def values(d: "dict[str, int]"):
    total = 0
    for k in d:
        total += d[k]
    yield total


print(list(values({"a": 1, "b": 2})))


def from_a_set(s: "set[int]"):
    out: "list[int]" = []
    for v in s:
        out.append(v * 10)
    yield sorted(out)


print(list(from_a_set({1, 2, 3})))


def built_inside():
    d = {"x": 1, "y": 2}
    names: "list[str]" = []
    for k in d:
        names.append(k)
    yield names


print(list(built_inside()))


def materialized(d: "dict[str, int]"):
    ks = list(d)
    for k in ks:
        yield k


print(list(materialized({"a": 1, "b": 2})))
