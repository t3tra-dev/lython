# Why execution: both are exact-text mimicry of CPython 3.14 raised at run
# time, so only running them shows a difference.
#
#   - math.exp overflow: CPython's math_1 checks errno after libm and raises
#     OverflowError("math range error") (Modules/mathmodule.c, is_error).
#     This returned inf.
#   - ord() on a str that is not one character: the message names the length
#     ("...but string of length 3 found"). This stopped at "expected a
#     character", which cannot tell empty from too-long.
import math


def main() -> None:
    print(math.exp(0.0), math.exp(1.0))
    try:
        print(math.exp(1000.0))
    except OverflowError as e:
        print(e)
    print(ord("A"), ord("z"))
    try:
        ord("abc")
    except TypeError as e:
        print(e)
    try:
        ord("")
    except TypeError as e:
        print(e)


main()
