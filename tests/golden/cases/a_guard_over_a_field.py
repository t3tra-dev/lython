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
# ⭐ AND A FACT DIES WHEN A NESTED STATEMENT ASSIGNS THE FIELD. The walk that
# erases one looked only at the statement's own targets, so
#
#     if c.f is not None:
#         for i in range(2):
#             if i == 1:
#                 c.f = None
#         print(c.f is None)
#
# still carried the proof into the read and raised, for a program CPython
# answers True. It looks inside a compound statement now -- but not into a
# nested def, lambda or class, which do not run there.
#
# Why this must run: every line below is a value, and the defects were silent
# (a crash, a lowering message about a binding nobody wrote, and a raise where
# CPython answers).


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



class Slot2:
    def __init__(self, v: "int | None") -> None:
        self.f = v


def cleared_in_a_loop() -> str:
    c = Slot2(5)
    if c.f is not None:
        for i in range(2):
            if i == 1:
                c.f = None
        return "none" if c.f is None else "value"
    return "-"


def cleared_in_a_nested_if() -> str:
    c = Slot2(5)
    c.f = 7
    if c.f is not None:
        c.f = None
    return "none" if c.f is None else "value"


def survives_an_unrelated_loop() -> int:
    c = Slot2(5)
    if c.f is not None:
        total = 0
        for i in range(3):
            total += i
        return c.f + total
    return -1


print(cleared_in_a_loop(), cleared_in_a_nested_if(), survives_an_unrelated_loop())


# ⭐ AND A UNION FIELD OF TWO REAL MEMBERS, which was gated off: narrowing an
# `int | str` field to `str` used to reach the ownership verifier, and the
# else arm of such a guard still read the whole union. Both were measured
# against the dead value a field starts at, and that is what was actually
# wrong; with it repaired the guard proves what it says on both sides.
class Payload:
    def __init__(self, v: "int | str") -> None:
        self.v: "int | str" = v

    def render(self) -> str:
        if isinstance(self.v, str):
            return self.v.upper()
        return "num" + str(self.v + 1)


def render_free(p: Payload) -> str:
    if isinstance(p.v, str):
        return "s:" + p.v + str(len(p.v))
    return "i:" + str(p.v + 1)


def render_else(p: Payload) -> str:
    if isinstance(p.v, int):
        return "i:" + str(p.v * 2)
    else:
        return "s:" + p.v


class Three:
    def __init__(self, v: "int | str | None") -> None:
        self.v: "int | str | None" = v

    def tell(self) -> str:
        if isinstance(self.v, str):
            return "s" + str(len(self.v))
        elif isinstance(self.v, int):
            return "i" + str(self.v + 1)
        return "n"


class Left:
    def __init__(self) -> None:
        self.k: int = 1


class Right:
    def __init__(self) -> None:
        self.k: int = 2


class Either:
    def __init__(self, v: "Left | Right") -> None:
        self.v: "Left | Right" = v

    def tell(self) -> str:
        if isinstance(self.v, Right):
            return "R" + str(self.v.k)
        return "L" + str(self.v.k)


def a_union_field_under_a_chain(p: Payload) -> str:
    if isinstance(p.v, str) and len(p.v) > 1:
        return p.v + "!"
    if not isinstance(p.v, str) or len(p.v) > 0:
        return "other"
    return "-"


def a_union_field_in_a_loop(p: Payload) -> int:
    total = 0
    for _ in range(3):
        if isinstance(p.v, str):
            total += len(p.v)
        else:
            total += p.v
    return total


def union_fields() -> None:
    print(Payload("ab").render(), Payload(4).render())
    print(render_free(Payload("xy")), render_free(Payload(4)))
    print(render_else(Payload("xy")), render_else(Payload(4)))
    print(Three("abc").tell(), Three(6).tell(), Three(None).tell())
    print(Either(Left()).tell(), Either(Right()).tell())
    print(a_union_field_under_a_chain(Payload("ab")))
    print(a_union_field_under_a_chain(Payload(3)))
    print(a_union_field_in_a_loop(Payload("ab")), a_union_field_in_a_loop(Payload(5)))


union_fields()
