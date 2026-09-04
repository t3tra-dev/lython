# A container built by a BUILTIN and stored in a field. The manifest answers
# with a contract NAME, and rebuilding a type from a name drops the parameters:
# `list(xs)` over a `list[str]` came out as a bare `builtins.list`, and the
# store was "attribute value builtins.list is not assignable to field
# builtins.list<str>". The manifest METHOD path already kept them -- `xs.copy()`
# in the same constructor worked -- so this was one question answered by two
# paths.


class Snapshot:
    ITEMS: list[str] = ["seed"]

    def __init__(self, xs: list[str], d: dict[str, int]) -> None:
        self.copied = list(xs)
        self.seeded = list(Snapshot.ITEMS)
        self.mapped = dict(d)
        self.uniq = set(xs)
        self.frozen = tuple(xs)
        self.ordered = sorted(xs)
        self.method_copy = xs.copy()

    def sizes(self) -> list[int]:
        return [
            len(self.copied),
            len(self.seeded),
            len(self.mapped),
            len(self.uniq),
            len(self.frozen),
            len(self.ordered),
            len(self.method_copy),
        ]

    def first(self) -> str:
        return self.copied[0] + self.seeded[0] + self.ordered[0]


class Chained:
    # A field's initializer may read an EARLIER field. `self.parts` resolves
    # through the protocol table, which learned this class's fields only after
    # the whole walk -- so `frozen` typed as `object` and joining it was
    # "cannot adapt builtins.object".
    def __init__(self, parts: tuple[str, ...]) -> None:
        self.parts = list(parts)
        self.frozen = tuple(self.parts)
        self.count = len(self.frozen)

    def show(self) -> str:
        return "".join(self.parts) + "|" + "".join(self.frozen) + str(self.count)


s = Snapshot(["b", "a"], {"k": 1})
print(s.sizes())
print(s.first(), s.frozen, sorted(s.uniq))
s.copied.append("c")
print(s.copied, Snapshot.ITEMS)

c = Chained(("x", "y"))
print(c.show())
