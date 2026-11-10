# isinstance and a class pattern over a union that holds a class beside its
# own subclass (`Wide | Cell`, `Wide(Cell)`) answer for both members: the
# subclass's own member, and the base member holding a subclass instance. The
# true side reads the value as the class tested, in a local, a field, a module
# global and a match; the false side keeps what the test left. Two subclasses
# of a pattern's class are both matched by it, and a class value whose classes
# lie below a member is matched as `isinstance` matches it, as is a tuple
# of classes one of which lies below a member.
import unittest


class Cell:
    def __init__(self, n: int) -> None:
        self.n = n

    def name(self) -> str:
        return "cell" + str(self.n)


class Wide(Cell):
    def __init__(self, n: int, w: int) -> None:
        super().__init__(n)
        self.w = w

    def width(self) -> int:
        return self.w * 10


class Other:
    def name(self) -> str:
        return "other"


def as_cell(n: int) -> Cell:
    return Wide(n, n + 1) if n % 2 == 0 else Cell(n)


def pick(n: int) -> Wide | Cell:
    if n % 3 == 0:
        return Wide(n, 7)
    return as_cell(n)


def three(n: int) -> Wide | Cell | Other:
    if n % 5 == 0:
        return Other()
    return pick(n)


def in_a_local(n: int) -> str:
    x = pick(n)
    if isinstance(x, Wide):
        return "W" + str(x.width())
    return "C" + x.name()


def in_a_local_of_three(n: int) -> str:
    x = three(n)
    if isinstance(x, Wide):
        return "W" + str(x.width())
    if isinstance(x, Cell):
        return "C" + x.name()
    return "O" + x.name()


def either_of_two(n: int) -> str:
    v = three(n)
    return "y" if isinstance(v, (Wide, Other)) else "n"


def as_an_expression(n: int) -> str:
    return "W" if isinstance(pick(n), Wide) else "other"


class Holder:
    def __init__(self, n: int) -> None:
        self.one: Wide | Cell = pick(n)

    def show(self) -> str:
        if isinstance(self.one, Wide):
            return "W" + str(self.one.width())
        return "-"


def matched(n: int) -> str:
    match three(n):
        case Wide() as w:
            return "W" + str(w.width())
        case Cell() as c:
            return "C" + c.name()
        case _:
            return "O"


current: Wide | Cell = pick(1)


def through_a_global() -> list[int]:
    global current
    out: list[int] = []
    for n in range(5):
        current = pick(n)
        if isinstance(current, Wide):
            out.append(current.width())
    return out


class Closer:
    def close(self) -> str:
        return "closed"


class File(Closer):
    pass


class Pipe(Closer):
    def close(self) -> str:
        return "pipe-closed"


class Rock:
    pass


def thing(n: int) -> File | Pipe | Rock:
    if n == 0:
        return File()
    if n == 1:
        return Pipe()
    return Rock()


def closed_by_a_pattern(n: int) -> str:
    match thing(n):
        case Closer() as c:
            return c.close()
        case _:
            return "rock"


class A:
    def __init__(self) -> None:
        self.a = 1

    def who(self) -> str:
        return "A"


class B(A):
    def __init__(self) -> None:
        super().__init__()
        self.b = 22

    def who(self) -> str:
        return "B" + str(self.b)


class C:
    pass


def made(n: int) -> A | B | C:
    if n == 0:
        return A()
    if n == 1:
        return B()
    return C()


def kept_at_their_base(n: int) -> str:
    x = made(n)
    if isinstance(x, A):
        return x.who() + str(x.a)
    return "C"


class Cmd:
    pass


class Go(Cmd):
    pass


class Say:
    pass


def handled(c: Cmd | Say, k: type[Go] | type[Say]) -> str:
    match c:
        case k():
            return "kind"
        case _:
            return "other"


def handled_after_a_class(c: Cmd | Say, k: type[Go] | type[Say]) -> str:
    match c:
        case Go():
            return "go"
        case k():
            return "kind"
        case _:
            return "other"


class AClassBesideItsSubclass(unittest.TestCase):
    def test_a_local(self) -> None:
        self.assertEqual([in_a_local(n) for n in range(7)],
                         ["W70", "Ccell1", "W30", "W70", "W50", "Ccell5",
                          "W70"])
        self.assertEqual([in_a_local_of_three(n) for n in range(11)],
                         ["Oother", "Ccell1", "W30", "W70", "W50", "Oother",
                          "W70", "Ccell7", "W90", "W70", "Oother"])
        self.assertEqual([as_an_expression(n) for n in range(3)],
                         ["W", "other", "W"])

    def test_a_tuple_of_classes(self) -> None:
        self.assertEqual([either_of_two(n) for n in range(6)],
                         ["y", "n", "y", "y", "y", "y"])

    def test_a_field_and_a_global(self) -> None:
        self.assertEqual([Holder(n).show() for n in range(7)],
                         ["W70", "-", "W30", "W70", "W50", "-", "W70"])
        self.assertEqual(through_a_global(), [70, 30, 70, 50])

    def test_a_class_pattern(self) -> None:
        self.assertEqual([matched(n) for n in range(11)],
                         ["O", "Ccell1", "W30", "W70", "W50", "O", "W70",
                          "Ccell7", "W90", "W70", "O"])

    def test_two_subclasses_of_the_class_tested(self) -> None:
        self.assertEqual([closed_by_a_pattern(n) for n in range(3)],
                         ["closed", "pipe-closed", "rock"])
        self.assertEqual([kept_at_their_base(n) for n in range(3)],
                         ["A1", "B221", "C"])

    def test_a_class_value_pattern(self) -> None:
        self.assertEqual(
            [handled(Say(), Say), handled(Go(), Say), handled(Go(), Go),
             handled(Cmd(), Go)],
            ["kind", "other", "kind", "other"])
        self.assertEqual(
            [handled_after_a_class(Say(), Say),
             handled_after_a_class(Go(), Say),
             handled_after_a_class(Go(), Go),
             handled_after_a_class(Cmd(), Go)],
            ["kind", "go", "go", "other"])


if __name__ == "__main__":
    unittest.main()
