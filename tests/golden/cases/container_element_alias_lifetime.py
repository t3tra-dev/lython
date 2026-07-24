# A container literal COPIES a reference to its element; it does not consume
# the source binding's. Storing a local into a tuple/list/dict used to hand
# the local's only claim to the container, so the local dangled the moment
# the container died -- and the next read silently printed the empty string
# instead of the value.
s = "hello"
t = (s,)
print(t, s)
print(s)
print(len(s))

runtime = "hel" + "lo"
pair = (runtime,)
print(pair)
print(runtime)
print(len(runtime))

items = [runtime]
print(items)
print(runtime)

mapping = {"k": runtime}
print(mapping)
print(runtime)
print(mapping["k"], runtime)

# The same value filling several slots hands over one token, not one per slot.
twice = (runtime, runtime)
print(twice)
print(runtime)

# Element read back out of a dead container still holds its own claim.
held = [runtime]
taken = held[0]
print(taken)
print(len(taken))


def local_scope(a: str, b: str) -> None:
    joined = a + b
    packed = (joined, a)
    print(packed)
    print(joined)
    print(a, b)


local_scope("fo", "od")

# Pure temporaries keep working: nothing else references them, so the
# container takes over their token.
print(["pear", "apple", "zebra"])
print(max(["pear", "apple", "zebra"]))
print({"x": "one", "y": "two"})


# A freshly constructed class instance stored as a dict VALUE reads back
# intact. Construction is `py.new` feeding `py.init`, so the instance has a
# second user besides the literal -- taking its token at the literal left the
# object with no claim at all and every field read after the lookup answered
# zero, with no diagnostic.
class P:
    def __init__(self, v: int) -> None:
        self.v = v


class Named:
    def __init__(self, n: str, v: int) -> None:
        self.n = n
        self.v = v


pairs = {"x": P(9)}
q = pairs["x"]
print(q.v)

multi = {"a": P(1), "b": P(2), "c": P(3)}
print(multi["a"].v, multi["b"].v, multi["c"].v)

named = {"k": Named("hello", 42)}
r = named["k"]
print(r.n, r.v)

# The same instance through the incremental (runtime-probe) dict path.
built: dict[str, P] = {}
built["z"] = P(77)
print(built["z"].v)

# ... and through a list, and through a named local that outlives the dict.
print([P(9)][0].v)
kept = P(5)
holder = {"p": kept}
print(holder["p"].v)
print(kept.v)
