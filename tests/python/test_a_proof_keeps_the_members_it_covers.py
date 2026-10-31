# A test that proves a name is not None keeps every other member: a walrus
# in an `if` inside a `for`, at module level or in a function, read after
# the loop too; a name stored by a loop at module level; a member that is a
# subclass of another. A comprehension filter with a walrus types the list
# it builds, and the walrus's name is then what a later `for` binds it to.
import unittest


class A:
    def show(self) -> str:
        return "a"


class B:
    def show(self) -> str:
        return "b"


class Base:
    def show(self) -> str:
        return "base"


class Derived(Base):
    def show(self) -> str:
        return "derived"


class Entry:
    def __init__(self, n: int) -> None:
        self.n = n


def get(i: int) -> A | B | None:
    if i == 0:
        return None
    return A() if i == 1 else B()


def get_related(i: int) -> Base | Derived | None:
    if i == 0:
        return None
    return Base() if i == 1 else Derived()


def maybe(i: int) -> int | None:
    return i if i != 0 else None


def entry(i: int) -> Entry | None:
    return Entry(i) if i % 2 == 0 else None


shown: list[str] = []
for i in [0, 1, 2]:
    if (z := get(i)) is not None:
        shown.append(z.show())
LAST_SHOWN = type(z).__name__

stored: list[str] = []
for i in [0, 1, 2]:
    w = get_related(i)
    if w is not None:
        stored.append(w.show())
LAST_STORED = type(w).__name__


def walrus_in_a_loop() -> list[str]:
    out: list[str] = []
    for i in [0, 1, 2]:
        if (z := get(i)) is not None:
            out.append(z.show())
    return out


def present(xs: list[int]) -> list[int]:
    return [y for x in xs if (y := maybe(x)) is not None]


def rebound() -> list[int]:
    xs = [e for i in range(5) if (e := entry(i)) is not None]
    out: list[int] = []
    for e in xs:
        out.append(e.n)
    return out


def upper(text):
    return text.upper()


def name(i: int) -> str | None:
    return "ab" if i == 0 else None


class AProofKeepsTheMembers(unittest.TestCase):
    def test_a_walrus_in_a_loop_at_module_level(self) -> None:
        self.assertEqual(shown, ["a", "b"])
        self.assertEqual(LAST_SHOWN, "B")

    def test_a_name_stored_by_a_loop_holding_a_subclass(self) -> None:
        self.assertEqual(stored, ["base", "derived"])
        self.assertEqual(LAST_STORED, "Derived")

    def test_a_walrus_in_a_loop_in_a_function(self) -> None:
        self.assertEqual(walrus_in_a_loop(), ["a", "b"])

    def test_a_comprehension_filter_with_a_walrus(self) -> None:
        self.assertEqual(present([0, 1, 2]), [1, 2])
        self.assertEqual(rebound(), [0, 2, 4])

    def test_an_unannotated_call_with_a_narrowed_walrus(self) -> None:
        if (line := name(0)) is not None:
            self.assertEqual(upper(line), "AB")


if __name__ == "__main__":
    unittest.main()
