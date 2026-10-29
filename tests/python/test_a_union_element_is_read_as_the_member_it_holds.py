# A container element whose type is a union is read as the member it holds,
# however it is reached: a constant or a run-time index into a display, a
# `pop`, an iteration, a generator frame that keeps it across a yield, and a
# name a loop rebinds to it.
import unittest
from typing import Iterator


class A:
    def __init__(self, n: int) -> None:
        self.n = n


class B:
    pass


def at(k: int) -> str:
    r = (1, "a")
    v = r[k]
    return type(v).__name__ + ":" + str(v)


def through_a_loop(k: int) -> list[str]:
    r = (A(k), str(k) + "y")
    out: list[str] = []
    for i in range(2):
        v = r[i]
        if isinstance(v, A):
            out.append("A" + str(v.n))
        else:
            out.append(v)
    return out


def undo(stack: list[int | str]) -> list[str]:
    out: list[str] = []
    while stack:
        v = stack.pop()
        if isinstance(v, int):
            out.append("i" + str(v))
        else:
            out.append("s" + v)
    return out


def kept_across_a_yield(v: int | None) -> Iterator[str]:
    yield str(v)
    yield repr(v)


def each_arm(v: int | str) -> Iterator[str]:
    yield type(v).__name__
    if isinstance(v, int):
        yield str(v + 1)
    else:
        yield v.upper()


def length_or_zero(flag: bool):
    held: str | None = "bb" if flag else None
    if held is not None:
        return len(held)
    else:
        return 0


def last_of(r: tuple[int, str]) -> str:
    j = 0
    last = None
    while j < 2:
        last = r[j]
        j += 1
    return repr(last)


class UnionElementTest(unittest.TestCase):
    def test_classes_read_by_a_constant_index(self):
        xs: list[type[int] | type[str]] = [int, str]
        c = xs[0]
        self.assertEqual((c is int, c.__name__, xs[1]("5")), (True, "int", "5"))
        kinds: list[type[A] | type[B]] = [A, B]
        self.assertTrue(kinds[1] == B)

    def test_a_run_time_index_into_a_display(self):
        self.assertEqual([at(0), at(1)], ["int:1", "str:a"])
        self.assertEqual(through_a_loop(3), ["A3", "3y"])
        ks = (A, B)
        self.assertEqual([ks[i].__name__ for i in range(2)], ["A", "B"])

    def test_pop(self):
        xs: list[int | None] = [1, None, 3]
        self.assertEqual([xs.pop(), xs.pop(), xs], [3, None, [1]])
        self.assertEqual(undo([1, "a", 2]), ["i2", "sa", "i1"])

    def test_iteration_of_ints_beside_a_bool_and_of_nones(self):
        self.assertEqual([type(v).__name__ for v in (1, True)],
                         ["int", "bool"])
        self.assertEqual(sum((1, True, 3)), 5)
        self.assertEqual([x for x in [None, None]], [None, None])

    def test_a_generator_frame(self):
        self.assertEqual([list(kept_across_a_yield(1)),
                          list(kept_across_a_yield(None))],
                         [["1", "1"], ["None", "None"]])
        self.assertEqual([list(each_arm(4)), list(each_arm("q"))],
                         [["int", "5"], ["str", "Q"]])

    def test_a_return_under_a_guard_is_typed_where_it_stands(self):
        self.assertEqual(length_or_zero(True) + length_or_zero(False), 2)

    def test_a_name_a_loop_rebinds(self):
        self.assertEqual(last_of((1, "a")), "'a'")

    def test_an_empty_list_beside_a_full_one(self):
        self.assertEqual([sum(xs) for xs in ([1, 2], [])], [3, 0])


if __name__ == "__main__":
    unittest.main()
