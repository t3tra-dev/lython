# A name bound to functions whose parameters differ, on paths that meet --
# an `if`, a `match`, a loop's back edge, a conditional expression, a list --
# calls whichever function it holds, as long as every one of them takes the
# call.
import unittest


def a(x: int) -> int:
    return x + 1


def b(x: int, y: int = 10) -> int:
    return x * y


def c(z: int) -> int:
    return z - 100


def d(*xs: int) -> int:
    return sum(xs) * 1000


def e(x: int, *, s: int = 7) -> int:
    return x * s


def by_if(n: int) -> int:
    k = a
    if n == 1:
        k = b
    return k(3)


def by_match(n: int) -> int:
    k = a
    match n:
        case 1:
            k = b
        case 2:
            k = c
        case _:
            pass
    return k(3)


def by_keyword(n: int) -> int:
    k = a
    if n == 1:
        k = b
    return k(x=3)


def by_kind(n: int) -> int:
    k = a
    if n == 1:
        k = d
    elif n == 2:
        k = e
    return k(3)


def by_conditional(n: int) -> int:
    k = b if n == 0 else c
    return k(3)


def by_lambda(n: int) -> int:
    k = b
    if n == 1:
        k = lambda x: x - 1
    return k(3)


def by_for(start: int) -> list[int]:
    out: list[int] = []
    k = a if start == 0 else b
    for i in range(3):
        out.append(k(i))
        k = c
    return out


def by_name_bound_in_the_body() -> list[int]:
    out: list[int] = []
    k = a
    for i in range(3):
        out.append(k(i))
        g = c
        k = g
    return out


def by_pair() -> list[int]:
    out: list[int] = []
    k = a
    for i in range(3):
        out.append(k(i))
        k, n = c, i
    return out


def by_while() -> list[int]:
    out: list[int] = []
    k = a
    i = 0
    while i < 3:
        out.append(k(i))
        k = b if i == 0 else c
        i += 1
    return out


module_k = a
module_results: list[int] = []
for module_i in range(3):
    match module_i:
        case 1:
            module_k = c
        case _:
            pass
    module_results.append(module_k(module_i))


class FunctionsOfDifferentShapesTest(unittest.TestCase):

    def test_an_if_and_a_match(self) -> None:
        self.assertEqual([by_if(0), by_if(1)], [4, 30])
        self.assertEqual([by_match(0), by_match(1), by_match(2)], [4, 30, -97])

    def test_a_keyword_both_name(self) -> None:
        self.assertEqual([by_keyword(0), by_keyword(1)], [4, 30])

    def test_star_args_and_keyword_only(self) -> None:
        self.assertEqual([by_kind(0), by_kind(1), by_kind(2)], [4, 3000, 21])

    def test_a_conditional_expression_and_a_lambda(self) -> None:
        self.assertEqual([by_conditional(0), by_conditional(1)], [30, -97])
        self.assertEqual([by_lambda(0), by_lambda(1)], [30, 2])

    def test_a_loop_back_edge(self) -> None:
        self.assertEqual(by_for(0), [1, -99, -98])
        self.assertEqual(by_for(1), [0, -99, -98])
        self.assertEqual(by_while(), [1, 10, -98])
        self.assertEqual(by_name_bound_in_the_body(), [1, -99, -98])
        self.assertEqual(by_pair(), [1, -99, -98])
        self.assertEqual(module_results, [1, -99, -98])

    def test_a_list(self) -> None:
        self.assertEqual([f(3) for f in [a, b, c]], [4, 30, -97])


if __name__ == "__main__":
    unittest.main()
