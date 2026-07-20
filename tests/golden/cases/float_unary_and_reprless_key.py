# float.__neg__/__pos__ manifest methods, and a KeyError for a key class
# without __repr__ must be catchable (default object repr, not an abort).

x = 1.5
print(-x, +x, -(-x))


def negate(v: float) -> float:
    return -v


print(negate(2.25), -0.0)


class K:
    def __init__(self, v: int) -> None:
        self.v = v

    def __hash__(self) -> int:
        return self.v

    def __eq__(self, other: "K") -> bool:
        return self.v == other.v


d = {K(1): "a"}
print(d[K(1)])
try:
    print(d[K(2)])
except KeyError:
    print("missing")
print("done")
