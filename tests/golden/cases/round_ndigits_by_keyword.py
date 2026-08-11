# Why execution: `round(x, ndigits=N)` COMPILED and printed the wrong number --
# the keyword was dropped and the call became `round(x)`. Only the value shows
# that, so the golden pins values for every spelling of the signature CPython
# documents as `round(number, ndigits=None)`.


def main() -> None:
    print(round(2.567, ndigits=1))
    print(round(2.567, 1))
    print(round(123.456, ndigits=2))
    print(round(1234.5678, ndigits=-2))
    print(round(number=2.5))
    print(round(2.5))
    print(round(2.567, None))
    print(round(1234, -2))
    print(round(1234, ndigits=-2))
    print(round(1234, ndigits=2))
    print(len([1, 2, 3]))


main()
