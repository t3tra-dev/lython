# What: the loop's iterator has no runtime object -- its position is a cell the
# suspension cannot carry -- so a generator that yields from a dict or set walk
# goes through the keys instead, and only running it shows the elements arrive
# in order and the whole walk completes.
def keys(d: "dict[str, int]"):
    for k in d:
        yield k


print(list(keys({"a": 1, "b": 2, "c": 3})))


def values(d: "dict[str, int]"):
    for k in d:
        yield d[k]


print(list(values({"a": 1, "b": 2})))


def views(d: "dict[str, int]"):
    for k in d.keys():
        yield k
    for v in d.values():
        yield str(v)
    for k, v in d.items():
        yield f"{k}={v}"


print(list(views({"a": 1, "b": 2})))


def elements(s: "set[int]"):
    for v in s:
        yield v * 10


print(sorted(elements({1, 2, 3})))


def stops_early(d: "dict[str, int]"):
    for k in d:
        if k == "b":
            break
        yield k


print(list(stops_early({"a": 1, "b": 2, "c": 3})))


def skips(d: "dict[str, int]"):
    for k in d:
        if d[k] % 2 == 0:
            continue
        yield k


print(list(skips({"a": 1, "b": 2, "c": 3})))


def writes_values(d: "dict[str, int]"):
    for k in d:
        d[k] = d[k] + 1
        yield d[k]


print(list(writes_values({"a": 1, "b": 2})))
