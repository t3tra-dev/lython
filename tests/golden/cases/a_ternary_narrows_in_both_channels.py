# What: the arms of `a if x is None else b` each see the narrowing their side
# of the test proves -- in the TYPE channel a pre-pass reads as well as in the
# emitted code. A class field is declared from that pre-pass, so only reading
# the field back shows the two agreed on what the ternary produces.
class Bag:
    def __init__(self, xs: "list[int] | None") -> None:
        self.xs = [] if xs is None else xs

    def size(self) -> int:
        return len(self.xs)


empty = Bag(None)
full = Bag([1, 2, 3])
print(empty.xs, full.xs, empty.size(), full.size())


class Named:
    def __init__(self, name: "str | None") -> None:
        self.name = "anon" if name is None else name

    def shout(self) -> str:
        return self.name.upper()


print(Named(None).shout(), Named("ann").shout())


class Table:
    def __init__(self, d: "dict[str, int] | None") -> None:
        self.d = {} if d is None else d

    def keys(self) -> "list[str]":
        return sorted(self.d)


print(Table(None).keys(), Table({"b": 2, "a": 1}).keys())


def coalesce(n: "int | None") -> int:
    return 0 if n is None else n + 1


print(coalesce(None), coalesce(4))


def reversed_test(n: "int | None") -> int:
    return n * 2 if n is not None else -1


print(reversed_test(None), reversed_test(3))
