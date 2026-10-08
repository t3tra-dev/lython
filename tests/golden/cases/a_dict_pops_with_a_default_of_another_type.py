# WHAT: dict.pop with a default of another type answers `V | T` as CPython's
# does -- `d.pop(k, None)` over a dict[int, int], over a field, in a loop, with
# the key and the default evaluated in order -- and pop(k) over a dict whose
# values are a union or `object` returns the value and removes it, raising
# KeyError(k) on a miss.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: the answers are the popped
# values and what is left in the dict; a pop that read and did not remove
# compiles and prints a dict still holding the key.

class Cache:
    def __init__(self) -> None:
        self.d: dict[str, list[int]] = {"a": [1, 2]}

    def take(self, k: str) -> list[int] | None:
        return self.d.pop(k, None)

c = Cache()
print(c.take("a"), c.take("a"), c.d)

def drain(d: dict[int, str], keys: list[int]) -> list[str]:
    out: list[str] = []
    for k in keys:
        v = d.pop(k, None)
        if v is None:
            out.append("-")
        else:
            out.append(v.upper())
    return out

print(drain({1: "a", 2: "b"}, [1, 3, 2, 1]))
m: dict[str, int | None] = {"x": None, "y": 3}
print(m.pop("x", None), m.pop("y", None), m.pop("z", None), m)
order: list[str] = []
def key() -> str:
    order.append("key")
    return "q"
def dflt() -> float:
    order.append("default")
    return 1.5
qd = {"q": 1}
print(qd.pop(key(), dflt()), order)
e = {}
print(e.pop("k", 3))
f: dict[str, int] = {}
print(f.pop("k", "none"))

m: dict[str, int | None] = {"x": None, "y": 30000, "z": 5}
y = m.pop("y")
if y is not None:
    print(y + 1)
print(m.pop("x") is None, m.popitem(), m)
words: dict[str, list[str] | str] = {"a": ["p" * 600, "q"], "b": "s" * 600}
for k in ["a", "b"]:
    v = words.pop(k)
    if isinstance(v, list):
        print(len(v[0]), v[1])
    else:
        print(len(v.upper()))
print(words)
big: dict[int, str] = {i: str(i) * 100 for i in range(50)}
got = [big.pop(i, None) for i in range(60)]
print(sum(len(g) for g in got if g is not None), got[-1], len(big))
try:
    m.pop("nope")
except KeyError as e:
    print(repr(e))





