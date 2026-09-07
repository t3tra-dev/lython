# Why execution: the answer is the VALUE the field has after the call, and the
# defect was a raise -- "attribute 'v' is None here, after a guard above proved
# it was not" -- for programs CPython runs to completion. A guard over a field
# is spent at the READ with a check, which is what makes carrying the proof
# safe; what was missing is that a call handed the OBJECT can rebind the field,
# so the proof does not reach past one.
#
# ⭐ And only what the callee is SEEN to assign counts. Erasing on every call
# was measured first and costs more than it buys: a call that assigns nothing
# would leave the read on the union, and `c.v + 1` on an `int | None` is
# refused -- so a logging call would take the narrowing away from the line
# after it. What the callee writes is asked too: a setter that stores the
# member the guard proved leaves the proof true.
class Cell:
    def __init__(self, v: "int | None") -> None:
        self.v: "int | None" = v
        self.w: "int | None" = v

    def clear(self) -> None:
        self.v = None

    def note(self) -> None:
        print("note")


class Sub(Cell):
    pass


def clears(c: Cell) -> None:
    c.v = None


def clears_at_index(n: int, c: Cell) -> None:
    c.v = None


def clears_in_a_branch(c: Cell) -> None:
    for i in range(1):
        if i == 0:
            c.v = None


def forwards(c: Cell) -> None:
    clears(c)


def sets_an_int(c: Cell) -> None:
    c.v = 3


def touches_another_field(c: Cell) -> None:
    c.w = None


def logs(c: Cell) -> None:
    print("log")


def main() -> None:
    a = Cell(5)
    if a.v is not None:
        clears(a)
        print(a.v)

    b = Cell(5)
    if b.v is not None:
        b.clear()
        print(b.v)

    c = Cell(5)
    if c.v is not None:
        clears_at_index(1, c)
        print(c.v)

    d = Cell(5)
    if d.v is not None:
        clears_in_a_branch(d)
        print(d.v)

    e = Cell(5)
    if e.v is not None:
        forwards(e)
        print(e.v)

    f = Cell(5)
    if f.v is not None:
        clears(c=f)
        print(f.v)

    g = Sub(5)
    if g.v is not None:
        g.clear()
        print(g.v)

    # The proof SURVIVES these: a call that writes the member the guard proved,
    # one that writes another field, one that writes nothing, and one made on a
    # different object of the same class.
    h = Cell(5)
    if h.v is not None:
        sets_an_int(h)
        print(h.v + 1)

    i = Cell(5)
    if i.v is not None:
        touches_another_field(i)
        print(i.v + 1)

    j = Cell(5)
    if j.v is not None:
        logs(j)
        print(j.v + 1)

    k = Cell(5)
    other = Cell(9)
    if k.v is not None:
        clears(other)
        other.clear()
        print(k.v + 1, other.v)

    # And the read BEFORE the call still has it.
    m = Cell(5)
    if m.v is not None:
        print(m.v + 1)
        clears(m)
        print(m.v)


main()
