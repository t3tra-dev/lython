# What a call hands a parameter that CPython takes as it is: `sep=None` and
# `end=None` are print's defaults, a `str | None` forwarded to them is either;
# an `object` parameter's default is the value written; a bool is an int for
# `*args: int`; and a lambda that prints with a keyword is the None it
# returns.
import io
import unittest
from collections.abc import Callable


def joined(a: int, b: int, sep: str | None = None,
           end: str | None = None) -> str:
    out = io.StringIO()
    print(a, b, sep=sep, end=end, file=out)
    return out.getvalue()


def shown(value: object = "d") -> str:
    return str(value)


def counted(value: object = 5) -> str:
    return repr(value)


def several(a: object = 2.5, b: object = True, c: object = None) -> str:
    return repr(a) + repr(b) + repr(c)


def total(*xs: int) -> int:
    return len(xs) * 100 + sum(xs)


class Holder:
    def put(self, value: object = "k") -> str:
        return str(value)

    def tagged(self, a: int, *rest: int) -> str:
        return str(a) + ":" + str(sum(rest))


class KeywordAndDefaultTest(unittest.TestCase):
    def test_none_is_print_default(self):
        self.assertEqual(
            [joined(1, 2), joined(1, 2, "-"), joined(1, 2, None, "!\n")],
            ["1 2\n", "1-2\n", "1 2!\n"])
        out = io.StringIO()
        print(*[1, 2], sep=None, file=out)
        print("x", end=None, file=out)
        self.assertEqual(out.getvalue(), "1 2\nx\n")

    def test_an_object_parameter_takes_its_written_default(self):
        self.assertEqual([shown(), shown(3), counted(), counted("x"),
                          several(), Holder().put(), Holder().put(1)],
                         ["d", "3", "5", "'x'", "2.5TrueNone", "k", "1"])

    def test_a_bool_is_an_int_for_args(self):
        self.assertEqual([total(True), total(1, True), total(True, False, 5)],
                         [101, 202, 306])
        self.assertEqual(Holder().tagged(True, True, 2), "True:3")

    def test_a_lambda_that_prints_with_a_keyword(self):
        out = io.StringIO()
        show: Callable[[int], None] = lambda r: print(r, end="!\n", file=out)
        show(1)
        self.assertEqual(out.getvalue(), "1!\n")


if __name__ == "__main__":
    unittest.main()
