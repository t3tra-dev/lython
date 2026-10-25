# Multiplication and normalization of ints past the machine word answer what
# CPython's do: products of every shape of operand (one digit against many,
# many against one, many against many, digits that carry at every position),
# their signs, zero, and the digit count of a result whose top digits cancel.
# The expected products come from schoolbook multiplication of the operands'
# decimal digits, done with small ints only.
import unittest


def decimal_product(x: str, y: str) -> str:
    xs = [ord(c) - 48 for c in reversed(x)]
    ys = [ord(c) - 48 for c in reversed(y)]
    out = [0] * (len(xs) + len(ys))
    for i in range(len(xs)):
        carry = 0
        for j in range(len(ys)):
            t = out[i + j] + xs[i] * ys[j] + carry
            out[i + j] = t % 10
            carry = t // 10
        k = i + len(ys)
        while carry > 0:
            t = out[k] + carry
            out[k] = t % 10
            carry = t // 10
            k += 1
    while len(out) > 1 and out[-1] == 0:
        out.pop()
    return "".join([chr(48 + d) for d in reversed(out)])


def all_ones(digits: int) -> int:
    # Every 30-bit digit is 2**30 - 1: each product digit carries.
    return 2 ** (30 * digits) - 1


def mixed(digits: int) -> int:
    # Digits that differ from each other, with a top digit of 1.
    return 2 ** (30 * digits - 30) + (7 ** (9 * digits)) % (2 ** (30 * digits - 30))


SIZES = [1, 2, 3, 35, 71]


class MulTest(unittest.TestCase):

    def check(self, a: int, b: int) -> None:
        expected = decimal_product(str(a), str(b))
        self.assertEqual(str(a * b), expected)
        self.assertEqual(str(b * a), expected)

    def test_carry_at_every_digit(self) -> None:
        for m in SIZES:
            for n in SIZES:
                self.check(all_ones(m), all_ones(n))

    def test_mixed_digits(self) -> None:
        for m in SIZES:
            for n in SIZES:
                self.check(mixed(m), mixed(n))

    def test_one_digit_times_many(self) -> None:
        for n in SIZES:
            self.check(mixed(n), 3)
            self.check(mixed(n), 2 ** 30 - 1)
            self.check(all_ones(n), 2 ** 29 + 12345)

    def test_signs(self) -> None:
        a = mixed(35)
        b = all_ones(3)
        p = a * b
        self.assertEqual(-a * b, -p)
        self.assertEqual(a * -b, -p)
        self.assertEqual(-a * -b, p)
        self.assertTrue(-a * b < 0)

    def test_zero(self) -> None:
        a = mixed(70)
        zero = len("")
        self.assertEqual(a * zero, 0)
        self.assertEqual(zero * a, 0)
        self.assertEqual(str(-a * zero), "0")

    def test_factorial_by_repeated_small_products(self) -> None:
        f = 1
        expected = "1"
        for i in range(1, 400):
            f *= i
            expected = decimal_product(expected, str(i))
        self.assertEqual(str(f), expected)

    def test_a_result_whose_top_digits_cancel(self) -> None:
        # The sum's capacity is one digit past the operands; the difference
        # loses all but its lowest digit.
        a = all_ones(70)
        b = all_ones(70) - 5
        self.assertEqual(a - b, 5)
        self.assertEqual(str((a + 1) - a), "1")
        self.assertEqual((a * a) - (a * a - 1), 1)


class SplitMulTest(unittest.TestCase):
    # Past 70 digits (140 for a square) a product is split in halves; past
    # twice the other operand's size it is taken one slice at a time. Each
    # product is checked by dividing it back, and a square against the product
    # of the int with an equal int that is not the same one.

    def check_by_division(self, a: int, b: int) -> None:
        p = a * b
        self.assertEqual(p // b, a)
        self.assertEqual(p % b, 0)
        self.assertEqual(p // a, b)
        self.assertEqual((-a) * b, -p)

    def test_balanced_sizes(self) -> None:
        for m, n in [(71, 71), (71, 100), (150, 200), (400, 401)]:
            self.check_by_division(mixed(m), mixed(n))
            self.check_by_division(all_ones(m), all_ones(n))

    def test_lopsided_sizes(self) -> None:
        for m, n in [(71, 142), (71, 500), (100, 1000)]:
            self.check_by_division(mixed(m), mixed(n))
            self.check_by_division(all_ones(m), all_ones(n))
            self.check_by_division(mixed(n), mixed(m))

    def test_a_slice_of_zero_digits(self) -> None:
        # The middle slices of the longer operand are all zero digits.
        a = mixed(80)
        b = 2 ** (30 * 400) + 12345
        self.check_by_division(a, b)

    def test_square_agrees_with_an_equal_distinct_int(self) -> None:
        for n in [71, 141, 150, 300, 700]:
            for a in [mixed(n), all_ones(n)]:
                b = (a + 1) - 1
                self.assertEqual(a * a, a * b)

    def test_power_agrees_with_repeated_products(self) -> None:
        base = mixed(80)
        p = 1
        for i in range(37):
            p = p * base
        self.assertEqual(base ** 37, p)

    def test_against_decimal_schoolbook_past_the_cutoff(self) -> None:
        a = mixed(72)
        b = all_ones(75)
        self.assertEqual(str(a * b), decimal_product(str(a), str(b)))


if __name__ == "__main__":
    unittest.main()
