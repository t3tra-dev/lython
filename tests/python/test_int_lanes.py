# An int read out of a list, a field, a dict or a closure, answered by an
# operator or returned by a runtime call, gives CPython's answer wherever it
# goes next: a conditional choice between two ints, a field store, a call, a
# return. Each case runs over small ints, ints that fill a machine word and
# ints wider than one; a bool held in a list where an int is declared stays
# the bool.
import unittest
from typing import Callable


BIG = 2 ** 70
WORD = 2 ** 62 + 5


def computed(values: list[int]) -> list[int]:
    # The same values, built at run time so each slot holds the value itself
    # rather than a shared literal object.
    out: list[int] = []
    for v in values:
        out.append(v + len(out) - len(out))
    return out


def running_max(xs: list[int]) -> int:
    best = xs[0]
    for i in range(1, len(xs)):
        v = xs[i]
        if v > best:
            best = v
    return best


def running_min_index(xs: list[int]) -> int:
    best = xs[0]
    at = 0
    for i in range(1, len(xs)):
        v = xs[i]
        if v < best:
            best = v
            at = i
    return at


class ChoiceTest(unittest.TestCase):

    def test_running_max_over_small_ints(self) -> None:
        xs = computed([(i * 7919) % 1009 - 500 for i in range(300)])
        self.assertEqual(running_max(xs), 506)
        self.assertEqual(xs[running_min_index(xs)], -500)

    def test_running_max_over_ints_past_the_word(self) -> None:
        xs = computed([3, BIG, -BIG, WORD, BIG + 1, 7])
        self.assertEqual(running_max(xs), BIG + 1)
        self.assertEqual(running_min_index(xs), 2)

    def test_running_max_keeps_a_bool(self) -> None:
        xs: list[int] = [False, True, 1, 0]
        self.assertEqual(repr(running_max(xs)), "True")
        self.assertEqual(repr(xs[running_min_index(xs)]), "False")

    def test_max_and_min_of_two_operator_answers(self) -> None:
        for a, b in [(3, 4), (BIG, 5), (-BIG, BIG), (WORD, WORD // 3)]:
            self.assertEqual(max(a * 3, b * 5), a * 3 if a * 3 > b * 5 else b * 5)
            self.assertEqual(min(a + 1, b - 1), a + 1 if a + 1 < b - 1 else b - 1)

    def test_a_choice_used_twice_as_an_object(self) -> None:
        xs = computed([BIG, 300, 2])
        best = xs[0]
        for v in xs:
            if v < best:
                best = v
        self.assertEqual((str(best), repr(best)), ("2", "2"))


class Counter:
    def __init__(self, start: int) -> None:
        self.n = start
        self.step = 3
        self.seen = start

    def doubled(self) -> int:
        return self.n * 2


class FieldTest(unittest.TestCase):

    def test_arithmetic_on_fields(self) -> None:
        for start in [0, 997, WORD, BIG, -BIG]:
            c = Counter(start)
            for _ in range(5):
                c.n = (c.n + c.step) % 1000003
            self.assertEqual(c.n, (start + 15) % 1000003)

    def test_a_field_stored_into_another(self) -> None:
        for start in [5, WORD, BIG]:
            c = Counter(start)
            d = Counter(0)
            d.seen = c.n
            c.n = 1
            self.assertEqual((d.seen, c.n), (start, 1))

    def test_a_field_read_once_and_twice_as_an_object(self) -> None:
        for start in [5, 300, BIG]:
            c = Counter(start)
            self.assertEqual(str(c.n), str(start))
            self.assertEqual(f"{c.n}:{c.n}", f"{start}:{start}")

    def test_a_field_read_in_a_loop(self) -> None:
        c = Counter(BIG)
        out: list[int] = []
        for _ in range(3):
            out.append(c.n)
        self.assertEqual(out, [BIG, BIG, BIG])

    def test_a_choice_between_fields(self) -> None:
        for a, b in [(1, 2), (BIG, WORD), (-BIG, 0)]:
            c = Counter(a)
            c.seen = b
            pick = c.n if c.n > c.seen else c.seen
            self.assertEqual(pick, a if a > b else b)


def make_scaler(k: int) -> Callable[[int], int]:
    def scale(x: int) -> int:
        return x * k + 1
    return scale


def twice(x: int) -> int:
    return x * 2


class CallTest(unittest.TestCase):

    def test_a_closure_over_an_int(self) -> None:
        for k in [3, WORD, BIG]:
            f = make_scaler(k)
            t = 0
            for _ in range(4):
                t = f(t) % 1000003
            expected = 0
            for _ in range(4):
                expected = (expected * k + 1) % 1000003
            self.assertEqual(t, expected)

    def test_an_int_out_of_a_dict_into_a_call(self) -> None:
        d = {1: 5, 2: WORD, 3: BIG}
        self.assertEqual([twice(d[k]) for k in [1, 2, 3]],
                         [10, 2 * WORD, 2 * BIG])

    def test_a_bool_out_of_a_list_into_a_call(self) -> None:
        xs: list[int] = [True, False]
        self.assertEqual((twice(xs[0]), twice(xs[1])), (2, 0))

    def test_an_int_returned_from_a_method(self) -> None:
        for start in [5, WORD, BIG]:
            c = Counter(start)
            self.assertEqual(c.doubled() - c.n, start)


class RuntimeAnswerTest(unittest.TestCase):

    def test_arithmetic_on_what_runtime_calls_answer(self) -> None:
        t = 0
        for w in ["banana", "kiwi", "apple", "cherry"]:
            t = t * 10 + w.find("a") + 1
        self.assertEqual(t, 2010)
        h = 0
        for c in "h\xe9llo":
            h = (h * 31 + ord(c)) % 1000003
        expected = 0
        for code in [104, 233, 108, 108, 111]:
            expected = (expected * 31 + code) % 1000003
        self.assertEqual(h, expected)

    def test_comparisons_with_what_runtime_calls_answer(self) -> None:
        self.assertTrue("abc".find("c") > len("a"))
        self.assertFalse(ord("a") >= ord("b"))
        self.assertEqual(int("12") * 5 + int("8"), 68)

    def test_an_answer_wider_than_the_word(self) -> None:
        big = int("1" * 30)
        self.assertEqual(big * 3 - big, big + big)
        self.assertTrue(int("9" * 25) > int("9" * 24) * 9)
        self.assertEqual((big + len("ab")) % 1000, 113)


if __name__ == "__main__":
    unittest.main()
