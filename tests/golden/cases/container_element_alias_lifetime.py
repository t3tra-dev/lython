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
