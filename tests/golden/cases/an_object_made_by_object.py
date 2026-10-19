# `object()` makes an object of class `object` and nothing else -- the
# sentinel CPython's unittest tells "no message" from `msg=None` by. Two of
# them are two objects: `is` and `==` are identity, the hash is the
# identity's, the repr names `object` and an address, and each lives as long
# as something holds it. It was refused ("cannot be constructed yet"):
# there was no allocation for it and no deallocator its release could reach.
# Why run it: identity is two allocations at run time, and the release of
# one through a box, a slot and a field is what the counts check.
_missing = object()


def lookup(table: dict[str, int], key: str, default: object = _missing) -> object:
    if key in table:
        return table[key]
    return default


a = object()
b = object()
print(a is b, a is a, a == b, a == a, a != b)
print(type(a).__name__, type(a) is object, isinstance(a, object))
print(hash(a) == hash(a), hash(a) == hash(b), bool(a))
r = repr(a)
print(r.startswith("<object object at 0x"), r.endswith(">"), str(a) == r)
xs: list[object] = [a, b, a]
print(xs.count(a), xs.index(b), b in xs, object() in xs)
seen: dict[object, str] = {a: "a", b: "b"}
print(seen[a], seen[b], len(seen))
print(lookup({"k": 1}, "k"), lookup({}, "k") is _missing)
print("[{}]".format(a).startswith("[<object object at 0x"))
kept = 0
for i in range(2000):
    o = object()
    group: list[object] = [o, a, o]
    if group.count(o) == 2 and o is not a and lookup({}, str(i)) is _missing:
        kept += 1
print(kept)
