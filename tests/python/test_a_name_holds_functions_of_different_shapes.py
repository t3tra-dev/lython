# A name bound to functions whose parameters differ, on paths that meet --
# an `if`, a `match`, a loop's back edge, a conditional expression, a list --
# calls whichever function it holds, as long as every one of them takes the
# call; and functions whose RESULTS differ answer with their own result.
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


def by_inner_loop_target() -> list[int]:
    out: list[int] = []
    k = a
    for i in range(3):
        out.append(k(i))
        for g in [c]:
            k = g
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


def half(x: int) -> float:
    return x / 2


def stars(x: int) -> str:
    return "*" * x


def nothing(x: int) -> None:
    return None


def result_by_if(n: int) -> int | float:
    k = a
    if n == 1:
        k = half
    return k(x=3)


def result_by_loop() -> list[int | float]:
    out: list[int | float] = []
    k = a
    for i in range(3):
        out.append(k(i))
        k = half
    return out


def result_by_lambda(n: int) -> int | float:
    base = 10
    k = a
    if n == 1:
        k = lambda x: x * 0.25 + base
    return k(4)


def result_decoded(n: int) -> str:
    k = a
    if n == 1:
        k = stars
    r = k(2)
    if isinstance(r, int):
        return "int " + str(r + 100)
    return "str " + r.upper()


def result_or_none(n: int) -> int | None:
    k = a
    if n == 1:
        k = nothing
    return k(1)


module_result_k = a
if len("ab") == 2:
    module_result_k = half


def read_module_result_k() -> int | float:
    return module_result_k(5)


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
        self.assertEqual(by_inner_loop_target(), [1, -99, -98])
        self.assertEqual(module_results, [1, -99, -98])

    def test_results_that_differ(self) -> None:
        self.assertEqual([result_by_if(0), result_by_if(1)], [4, 1.5])
        self.assertEqual(
            [type(result_by_if(n)).__name__ for n in range(2)], ["int", "float"]
        )
        self.assertEqual(result_by_loop(), [1, 0.5, 1.0])
        self.assertEqual([result_by_lambda(0), result_by_lambda(1)], [5, 11.0])
        self.assertEqual(
            [result_decoded(0), result_decoded(1)], ["int 103", "str **"]
        )
        self.assertEqual([result_or_none(0), result_or_none(1)], [2, None])
        self.assertEqual(module_result_k(5), 2.5)
        self.assertEqual(read_module_result_k(), 2.5)

    def test_a_list(self) -> None:
        self.assertEqual([f(3) for f in [a, b, c]], [4, 30, -97])


if __name__ == "__main__":
    unittest.main()
