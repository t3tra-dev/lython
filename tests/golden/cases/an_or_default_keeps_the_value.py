# What: `a or b` yields the operand that decided it, not a bool -- and an empty
# literal on either side has no element type to contribute. Only printing what
# came out shows which operand the expression produced.
def coalesce(xs: "list[int] | None") -> "list[int]":
    return xs or []


print(coalesce(None), coalesce([2, 1]))


class Bag:
    def __init__(self, xs: "list[int] | None") -> None:
        self.v = xs or []

    def size(self) -> int:
        return len(self.v)


print(Bag(None).v, Bag([1, 2]).v, Bag(None).size(), Bag([1, 2]).size())


def name_or(n: "str | None") -> str:
    return n or "anon"


print(name_or(None), name_or(""), name_or("ann"))


def first_of(a: "list[int]", b: "list[int]") -> "list[int]":
    return a or b


print(first_of([], [3]), first_of([1], [3]))


def table(d: "dict[str, int] | None") -> "dict[str, int]":
    return d or {}


print(sorted(table(None)), sorted(table({"a": 1})))


def either(a: "list[int]", b: "list[int]") -> "list[int]":
    return a and b


print(either([], [5]), either([2], [5]))
