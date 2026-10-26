# WHAT: an `@overload`ed function whose implementation returns a value the
# overload the call matched does not declare raises TypeError at the call,
# naming both types. (CPython hands the value on; here the call's type is the
# overload's, and no value of it can be the int.)
#
# WHY THIS IS RUN: the check is made on the value the implementation returned,
# which only the run has.
from typing import overload


@overload
def broken(x: int) -> str: ...
@overload
def broken(x: str) -> int: ...
def broken(x: int | str) -> int | str:
    return x


print(broken("ab"))
