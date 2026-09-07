# `if self.left is None:` proves something about a FIELD, and the machinery
# only knew names -- so the binary-search-tree idiom, and every linked
# structure with it, was refused: "union<Tree, None> does not provide manifest
# method 'insert'". `v = self.left; if v is not None:` -- the same program with
# the read bound first -- has always compiled.
#
# ⭐ AND THE PROOF SPENT INSIDE A SHORT-CIRCUIT CHAIN, which is how a running
# maximum is written:
#
#     if self.best is None or v > self.best:
#
# The chain bound the PATH as a local symbol, so `types.lookupSymbol("self.best")`
# answered and the attribute read took the qualified-module-symbol road:
# "unresolved runtime binding 'self.best'" out of the lowering. A path's proof
# is spent at the READ with a check, and it is recorded per OPERAND -- recording
# it for the whole chain made the read in the operand that PROVES it check a
# fact that does not hold yet, and it raised.
#
# ⭐ AND A WRITE UNDER SUCH A PROOF. `if s.best is not None: s.best = 3` stored
# `literal<3>` into a field whose lanes are a union -- the store did not coerce
# to the DECLARED type, because the value had already been typed by the
# narrowing -- and the program SEGFAULTED with the guard false, so the store
# never even ran. The `else` spelling and the same write with the field already
# set were both correct, which is what kept it hidden.
#
# Why this must run: every line below is a value, and both defects were silent
# (one a crash, one a lowering message about a binding nobody wrote).


class Tree:
    def __init__(self, v: int) -> None:
        self.v = v
        self.left = None
        self.right = None

    def insert(self, v: int) -> None:
        if v < self.v:
            if self.left is None:
                self.left = Tree(v)
            else:
                self.left.insert(v)
        else:
            if self.right is None:
                self.right = Tree(v)
            else:
                self.right.insert(v)

    def walk(self) -> list[int]:
        out: list[int] = []
        if self.left is not None:
            out += self.left.walk()
        out.append(self.v)
        if self.right is not None:
            out += self.right.walk()
        return out

    # The early-return spelling of the same guard: the fact survives the `if`
    # because only one side of it reaches the line below.
    def depth(self) -> int:
        if self.left is None:
            return 1
        return 1 + self.left.depth()


t = Tree(5)
for value in [3, 8, 1, 4, 9]:
    t.insert(value)
print(t.walk())
print(t.depth())


# The `isinstance` spelling of the same guard, over a field whose static type
# is a base class: the read is checked against the CLASS the guard proved and
# then refined, which is the same shape as the union test above.
class Shape:
    def area(self) -> int:
        return 0


class Square(Shape):
    sides = 4

    def __init__(self, n: int) -> None:
        self.n = n

    def area(self) -> int:
        return self.n * self.n


class Holder:
    def __init__(self, s: Shape) -> None:
        self.s = s

    def describe(self) -> str:
        if isinstance(self.s, Square):
            return "sq" + str(self.s.sides) + "/" + str(self.s.area())
        return "shape" + str(self.s.area())

    def sides(self) -> int:
        if not isinstance(self.s, Square):
            return -1
        return self.s.sides


print(Holder(Shape()).describe(), Holder(Square(3)).describe())
print(Holder(Shape()).sides(), Holder(Square(3)).sides())


class Box:
    def __init__(self) -> None:
        self.v = None

    def set(self, s: str) -> None:
        self.v = s

    # Two reads under one guard, and a conditional expression spelling.
    def shout(self) -> str:
        if self.v is not None:
            return self.v.upper() + self.v.lower()
        return "-"

    def sized(self) -> str:
        return str(len(self.v)) if self.v is not None else "-"

    # An Optional field narrowed by isinstance rather than by `is None`.
    def shouted(self) -> str:
        if isinstance(self.v, str):
            return self.v.title()
        return "-"


b = Box()
print(b.shout(), b.sized(), b.shouted())
b.set("Ab")
print(b.shout(), b.sized(), b.shouted())


class Stats:
    def __init__(self) -> None:
        self.best: "int | None" = None
        self.worst: "int | None" = None
        self.tag: "str | None" = None

    def add(self, v: int) -> None:
        if self.best is None or v > self.best:
            self.best = v
        if self.worst is not None and v < self.worst:
            self.worst = v
        elif self.worst is None:
            self.worst = v

    def label(self, t: str) -> None:
        if self.tag is None or t > self.tag:
            self.tag = t


def high_water(values: list[int]) -> "int | None":
    best: "int | None" = None
    for v in values:
        if best is None or v > best:
            best = v
    return best


class Slot:
    def __init__(self) -> None:
        self.v: "int | None" = None


def written_under_a_guard() -> None:
    s = Slot()
    # The guard is FALSE, so the body never runs -- and the store inside it
    # still has to be lowered against the declared union.
    if s.v is not None:
        s.v = 3
    print(s.v is None)
    s.v = 1
    if s.v is not None:
        s.v = 3
    print(s.v)


def main() -> None:
    stats = Stats()
    for v in [3, 7, 2, 7]:
        stats.add(v)
    stats.label("b")
    stats.label("a")
    stats.label("c")
    print(stats.best, stats.worst, stats.tag)
    print(high_water([]), high_water([4]), high_water([4, 9, 1]))
    written_under_a_guard()


main()
