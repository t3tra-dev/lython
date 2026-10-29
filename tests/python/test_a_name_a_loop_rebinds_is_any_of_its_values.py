# A name a loop rebinds holds, inside the loop, whatever any trip gave it --
# not the value it was first bound to. An index first bound to 0 and stepped
# in the loop reads every member of a tuple whose members differ, and what a
# generator built from them yields, or a list they fill holds, is typed for
# every member. So are the generators `zip` and `enumerate` are as values.
import unittest


def described(v: int | str) -> str:
    return v.upper() if isinstance(v, str) else str(v + 1)


def stepped_tuples(r: tuple[int, str]):
    i = 0
    while i < 2:
        out = []
        out.append(r[i])
        yield tuple(out)
        i = i + 1


def stepped_members(r: tuple[int, str]):
    i = 0
    while i < 2:
        yield r[i]
        i = i + 1


def filled(r: tuple[int, str]):
    i = 0
    out = []
    while i < 2:
        out.append(r[i])
        i = i + 1
    return out


def columns(rows: tuple[tuple[int, str], ...]):
    i = 0
    while True:
        out = []
        for r in rows:
            if i >= len(r):
                return
            out.append(r[i])
        yield tuple(out)
        i = i + 1


class LoopReboundNameTest(unittest.TestCase):
    def test_a_generator_yields_what_every_trip_reads(self):
        self.assertEqual([described(v) for t in stepped_tuples((1, "a"))
                          for v in t], ["2", "A"])
        self.assertEqual([described(v) for v in stepped_members((1, "a"))],
                         ["2", "A"])
        self.assertEqual([described(v) for t in columns(((1, "a"), (2, "b")))
                          for v in t], ["2", "3", "A", "B"])

    def test_a_list_the_loop_fills(self):
        self.assertEqual([described(v) for v in filled((1, "a"))], ["2", "A"])

    def test_zip_and_enumerate_as_values(self):
        z = zip((1, "a"), (2, "b"))
        self.assertEqual([described(p) + described(q) for p, q in z],
                         ["23", "AB"])
        e = enumerate((1, "a"))
        self.assertEqual([f"{i}{described(v)}" for i, v in e], ["02", "1A"])


if __name__ == "__main__":
    unittest.main()
