# Why execution: the wrong answer was a VALUE -- nan and -inf where CPython
# raises. The compiler exited 0 and printed a float, so only running and
# comparing tells them apart, and a caught ValueError is only observable at
# runtime.
#
# CPython's math_1 checks the operand and raises ValueError("math domain
# error"); returning the IEEE result the hardware produces is the one thing it
# does not do. log rejects the whole non-positive domain, so log(0.0) is a
# domain error too, not -inf.
import math


def main() -> None:
    print(math.sqrt(4.0))
    print(math.sqrt(0.0))
    print(round(math.log(2.718281828459045), 6))
    print(round(math.log(1.0), 6))

    try:
        print(math.sqrt(-1.0))
    except ValueError as err:
        print(str(err))

    try:
        print(math.log(0.0))
    except ValueError as err:
        print(str(err))

    try:
        print(math.log(-1.0))
    except ValueError as err:
        print(str(err))


main()
