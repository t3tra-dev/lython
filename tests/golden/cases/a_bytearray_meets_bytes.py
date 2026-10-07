# WHAT: a bytearray among other values -- equal to and ordered against bytes
# by content in either order, also when both sit boxed in containers (list
# equality, `in`, sorted over `bytes | bytearray`); bytes(ba) and the mixed
# + (bytes on the left gives bytes); printed inside lists, dicts and tuples;
# mutated through a dict entry; returned across a handler and out of a `with`.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: the boxed comparisons are
# decided at run time by the classes the boxes hold -- `[bytearray(b"a")] ==
# [b"a"]` answered False -- and the containers print what they hold then.

def grow(b: bytearray, n: int) -> bytearray:
    for i in range(n):
        b.append(65 + i % 26)
    return b
xs = [bytearray(b"a"), bytearray(b"bc")]
print(xs, len(xs[1]))
print([bytearray(b"a")] == [bytearray(b"a")], [bytearray(b"a")] == [b"a"], [b"a"] == [bytearray(b"a")])
d = {"k": bytearray(b"v")}
d["k"].extend(b"w")
print(d)
total = 0
for byte in bytearray(b"\x01\x02\x03"):
    total += byte
print(total)
grown = grow(bytearray(), 40)
print(len(grown), grown[:5], grown[-3:])
t = (bytearray(b"x"), 1)
print(t)
print(sorted([bytearray(b"b"), bytearray(b"a")]))
print(bytearray(b"abc") in [bytearray(b"abc")])
print(f"{bytearray(b'q')!r}", str(bytearray(b"z")))
xs: list[bytes | bytearray] = [b"b", bytearray(b"a"), b"c"]
print(sorted(xs), bytearray(b"a") in [b"x", b"a"], [b"a", bytearray(b"b")] == [bytearray(b"a"), b"b"])
class CM:
    def __enter__(self) -> "CM":
        return self
    def __exit__(self, a: object, b: object, c: object) -> None:
        pass
def f() -> bytearray:
    b = bytearray(b"x")
    try:
        b.append(49)
        raise ValueError("v")
    except ValueError:
        b.append(50)
    return b
def g() -> bytearray:
    with CM():
        return bytearray(b"w")
def h(n: int) -> bytearray:
    out = bytearray()
    for i in range(n):
        try:
            if i % 2 == 0:
                raise KeyError(i)
            out.append(48 + i)
        except KeyError:
            out += b"-"
    return out
print(f(), g(), h(6))
