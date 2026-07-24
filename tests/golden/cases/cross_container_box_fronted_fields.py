# Cross-track: wave25/defects made a container literal COPY its element's
# reference instead of consuming the source binding's, and wave25/abi moved a
# str field behind a stable box so a rebind is visible through a stored dict
# key. Both rewrite who owns a payload and when it is released, and they meet
# wherever a class instance with str fields is BOTH a container element and a
# dict key. Neither track's own case puts the two together.


class K:
    def __init__(self, v: str) -> None:
        self.v = v

    def __hash__(self) -> int:
        return len(self.v) * 7

    def __eq__(self, other: "K") -> bool:
        return self.v == other.v

    def __repr__(self) -> str:
        return "K(" + repr(self.v) + ")"


# A named local stored as a dict VALUE, then mutated through the local: the
# literal copied the reference, so the local still owns its own claim and the
# box-fronted field follows the rebind.
held = K("a")
values = {"one": held}
print(values["one"].v)
held.v = "b"
print(held.v)
print(values["one"].v)
print(len(values["one"].v))


# The same instance as a dict KEY and as a list element at once. Mutating the
# field after insertion leaves the key's stale hash but a live payload.
key = K("xy")
table = {}
table[key] = "stored"
bucket = [key]
key.v = "xyz"
print(bucket[0].v)
print(len(table))
# Both miss: the entry sits in the old hash's bucket but no longer compares
# equal to the old value, exactly as CPython's stale-hash entry behaves.
print(K("xy") in table, K("xyz") in table)
print(table[K("xy")] if K("xy") in table else "gone")
print(key.v)


# A runtime-built string into a field, into a container, and read back: the
# element read holds its own claim after the container dies.
built = "he" + "llo"
holder = K(built)
print(built)
pair = (holder, holder)
print(pair[0].v, pair[1].v)
print(built)
taken = [holder][0]
print(taken.v)
print(len(taken.v))


# Instances constructed straight into containers (no named local at all), then
# a field read after the lookup -- construction feeds `py.new` into `py.init`,
# so the literal must not take the instance's only token.
inline = {"a": K("one"), "b": K("two")}
print(inline["a"].v, inline["b"].v)
print([K("x"), K("y")][1].v)
print((K("t"),)[0].v)


# The incremental (runtime-probe) dict path with a field mutation in between.
grown: dict[str, K] = {}
grown["k"] = K("start")
print(grown["k"].v)
grown["k"].v = "moved"
print(grown["k"].v)


# A field read escaping into a container that outlives its owner.
def name_of(value: K) -> str:
    return value.v


names = [name_of(K("gone")), name_of(K("also"))]
print(names)
print(names[0], len(names[1]))
