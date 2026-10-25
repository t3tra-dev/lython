# Fixed-point and scientific formatting of floats, and round(), answer the
# correctly rounded decimal: the digits of the exact binary value, rounded
# half to even at the cutoff. The expected text comes from the float's exact
# ratio (as_integer_ratio) rounded with ints; the values cover the digit
# counts CPython takes without big integers (up to 14 digits, and integers
# below 10**15) and the ones it does not.
import unittest


def rounded_ratio(n: int, d: int) -> int:
    # n / d rounded half to even (d > 0).
    q, r = divmod(n, d)
    twice = 2 * r
    if twice > d or (twice == d and q % 2 == 1):
        q += 1
    return q


def exact_fixed(x: float, places: int) -> str:
    n, d = x.as_integer_ratio()
    negative = n < 0
    q = rounded_ratio(abs(n) * 10 ** places, d)
    digits = str(q)
    if places > 0:
        digits = digits.rjust(places + 1, "0")
        text = digits[:-places] + "." + digits[-places:]
    else:
        text = digits
    return ("-" if negative else "") + text


def exact_scientific(x: float, places: int) -> str:
    n, d = x.as_integer_ratio()
    negative = n < 0
    n = abs(n)
    # The decimal exponent of the leading digit: 10**k <= n/d < 10**(k+1).
    k = len(str(n // d)) - 1 if n >= d else -len(str(d // n))
    while n * 10 ** max(0, -k) < d * 10 ** max(0, k):
        k -= 1
    while n * 10 ** max(0, -k - 1) >= d * 10 ** max(0, k + 1):
        k += 1
    shift = places - k
    if shift >= 0:
        q = rounded_ratio(n * 10 ** shift, d)
    else:
        q = rounded_ratio(n, d * 10 ** (-shift))
    if q == 10 ** (places + 1):
        q = 10 ** places
        k += 1
    digits = str(q)
    mantissa = digits[0] + ("." + digits[1:] if places > 0 else "")
    sign = "-" if k < 0 else "+"
    exponent = str(abs(k)).rjust(2, "0")
    return ("-" if negative else "") + mantissa + "e" + sign + exponent


def sample_values() -> list[float]:
    values: list[float] = []
    for i in range(1, 120):
        m = (i * 7919) % 1000003
        for e in [-12, -5, -2, 0, 1, 3, 9, 14, 15, 16, 22]:
            values.append(m / 1000.0 * 10.0 ** e)
            values.append(-m / 7.0 * 10.0 ** e)
    for v in [0.125, 0.375, 2.675, 1.005, 0.5, 1.5, 2.5, 3.5, 1e15 + 0.5,
              999999999999999.9, 123456789012345.67, 9.995, 99.995, 0.045,
              1e22, 1e23, 0.1, 0.7, 1e-7, 123.456, 1234567.0, 100.0,
              999.0, 1e14, 1e16, 9999999999999998.0, 0.9999999999999999,
              5e-324, 2.2250738585072014e-308]:
        values.append(v)
    return values


class FloatFormatTest(unittest.TestCase):

    def test_fixed_point(self) -> None:
        for v in sample_values():
            for places in [0, 1, 2, 3, 6, 10, 14, 17]:
                self.assertEqual(format(v, "." + str(places) + "f"),
                                 exact_fixed(v, places))

    def test_scientific(self) -> None:
        for v in sample_values():
            if v == 0.0:
                continue
            for places in [0, 1, 2, 6, 13, 16]:
                self.assertEqual(format(v, "." + str(places) + "e"),
                                 exact_scientific(v, places))

    def test_round_to_places(self) -> None:
        for v in sample_values():
            for places in [-3, -1, 0, 1, 2, 5]:
                n, d = v.as_integer_ratio()
                if places >= 0:
                    q = rounded_ratio(abs(n) * 10 ** places, d)
                    expected = q / 10 ** places
                else:
                    q = rounded_ratio(abs(n), d * 10 ** (-places))
                    expected = float(q * 10 ** (-places))
                if n < 0:
                    expected = -expected
                self.assertEqual(round(v, places), expected)

    def test_halfway_values_round_to_even(self) -> None:
        self.assertEqual(format(0.125, ".2f"), "0.12")
        self.assertEqual(format(0.375, ".2f"), "0.38")
        self.assertEqual(format(2.5, ".0f"), "2")
        self.assertEqual(format(3.5, ".0f"), "4")
        self.assertEqual(format(2.675, ".2f"), "2.67")
        self.assertEqual(format(1e15 + 0.5, ".0f"), "1000000000000000")
        # Integers whose cut digit is an exact half.
        self.assertEqual(format(125.0, ".1e"), "1.2e+02")
        self.assertEqual(format(135.0, ".1e"), "1.4e+02")
        self.assertEqual(format(1250.0, ".1e"), "1.2e+03")
        self.assertEqual(round(500.0, -3), 0.0)
        self.assertEqual(round(501.0, -3), 1000.0)
        self.assertEqual(round(1500.0, -3), 2000.0)
        self.assertEqual(round(2500.0, -3), 2000.0)
        self.assertEqual(round(-2500.0, -3), -2000.0)

    def test_rounding_carries_into_a_new_digit(self) -> None:
        self.assertEqual(format(9.995, ".2f"), "9.99")
        self.assertEqual(format(9.9951, ".2f"), "10.00")
        self.assertEqual(format(999999.5, ".0f"), "1000000")
        self.assertEqual(format(9.96, ".1e"), "1.0e+01")


class IntFormatTest(unittest.TestCase):

    def test_an_empty_spec_is_str(self) -> None:
        self.assertEqual(format(12345, ""), "12345")
        self.assertEqual(format(-7, ""), "-7")
        self.assertEqual(format(2 ** 70, ""), "1180591620717411303424")
        self.assertEqual(f"{len('abc')}", "3")
        self.assertEqual(format(True, ""), "True")
        self.assertEqual(f"{False}", "False")


if __name__ == "__main__":
    unittest.main()
