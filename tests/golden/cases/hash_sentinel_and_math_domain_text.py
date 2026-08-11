# Why execution: both are exact-text / exact-value mimicry of CPython 3.14, so
# only running them against it shows a difference.
#
#   - a user __hash__ that returns -1: CPython remaps it to -2, because -1 is
#     its error sentinel (Objects/object.c, PyObject_Hash). This printed -1.
#   - math.sqrt/log domain errors: 3.14 replaced the one generic "math domain
#     error" with a per-function message that interpolates the operand
#     (gh-101410). These carried the 3.13 text.
import math


class Sentinel:
    def __hash__(self) -> int:
        return -1


class Ordinary:
    def __hash__(self) -> int:
        return 7


class Big:
    def __hash__(self) -> int:
        return -123456789


def main() -> None:
    print(hash(Sentinel()))
    print(hash(Ordinary()))
    print(hash(Big()))
    print(hash(-1), hash(7))
    try:
        print(math.sqrt(-1.0))
    except ValueError as e:
        print(e)
    try:
        print(math.log(0.0))
    except ValueError as e:
        print(e)
    try:
        print(math.log(-2.5))
    except ValueError as e:
        print(e)
    print(math.sqrt(4.0), math.log(1.0))


main()
