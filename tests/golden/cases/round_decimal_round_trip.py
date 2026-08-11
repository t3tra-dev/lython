# Why execution: every line is a number CPython computes one way and the
# binary scaling computed another. `round(x, n)` used to be
# `roundeven(x * 10**n) / 10**n`, which decides the digit in BINARY before the
# rounding rule is consulted -- one cause with three symptoms: manufactured
# ties (2.675 * 100 IS 267.5, so half-to-even went up where CPython says
# 2.67), a lost last digit (round(234743633112.0, 8) gave
# 234743633112.00003), and overflow at exponents CPython never reaches
# (round(1e300, 15) gave inf). CPython formats to `ndigits` decimal places and
# parses back; so does this now.


def large_magnitudes() -> None:
    print(round(1e300, 9))
    print(round(1e300, 15))
    print(round(-1e300, 9))
    print(round(1.7976931348623157e308, 1))


def digits_that_were_lost() -> None:
    print(round(234743633112.0, 8))
    print(round(123456.789, 3))
    print(round(0.1, 20))
    print(round(1e16, 2))


def ties_the_scaling_manufactured() -> None:
    print(round(2.675, 2))
    print(round(-2.675, 2))
    print(round(1.005, 2))
    print(round(2.567, 1))


def ties_without_ndigits() -> None:
    print(round(0.5), round(1.5), round(2.5), round(3.5))
    print(round(-0.5), round(-1.5), round(-2.5))


def extreme_ndigits() -> None:
    # CPython's NDIGITS_MAX / NDIGITS_MIN: past them the value passes through,
    # or becomes a zero that keeps its sign.
    print(round(1e300, -309))
    print(round(5e-324, 323))
    print(round(-1.0, -309))
    # PyNumber_AsSsize_t(o_ndigits, NULL) CLIPS rather than raising, so an
    # ndigits no ssize_t can hold is the extreme, not an OverflowError.
    print(round(1.5, 10**30))
    print(round(1.5, -(10**30)))


def negative_ndigits() -> None:
    print(round(1234.5678, -2), round(1234.5678, -3), round(1250.0, -2))


def main() -> None:
    large_magnitudes()
    digits_that_were_lost()
    ties_the_scaling_manufactured()
    ties_without_ndigits()
    extreme_ndigits()
    negative_ndigits()


main()
