# Why execution: the wrong answer was a digit. round(2.675, 2) returned 2.68
# where CPython returns 2.67, exit 0, no diagnostic -- only comparing the
# printed value against CPython tells them apart.
#
# 2.675 is really 2.67499999999999982, so the decimal answer is 2.67. But
# 2.675 * 100 rounds to exactly 267.5 in binary64, and half-to-even on a value
# that IS the midpoint goes up: the scaling manufactured a tie the original
# number does not have. A real tie (0.125 at two places, 2.5 at zero) must
# still go to even, and round(-0.5) must still be -0.0.


def main() -> None:
    print(round(2.675, 2))
    print(round(-2.675, 2))
    print(round(8.835, 2))
    print(round(-9.6335, 3))
    print(round(-16.15, 1))
    print(round(81.685, 2))

    print(round(0.125, 2))
    print(round(2.345, 2))
    print(round(1.005, 2))

    print(round(2.5))
    print(round(1.5))
    print(round(0.5))
    print(round(-0.5))
    print(round(-0.5, 0))

    print(round(3.14159, 3))
    print(round(1e16, 2))
    print(round(12345.678, -2))


main()
