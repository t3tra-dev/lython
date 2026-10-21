# WHAT: `if x is None: x = default` on a parameter whose type is a union of two
# classes and None, then the value passed where only the two classes are
# accepted -- with the default and with each class.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: the None stayed in the type
# after the `if` (a three-member union narrows to a union, not a member), so
# the call was refused; which class arrives is a run-time answer.
import io
import sys


class A:
    pass


class B:
    pass


def need(x: A | B) -> str:
    return type(x).__name__


def pick(x: A | B | None = None) -> str:
    if x is None:
        x = A()
    return need(x)


def stream(s: io.TextIOWrapper | io.StringIO | None = None) -> str:
    if s is None:
        s = sys.stdout
    return type(s).__name__


print(pick(), pick(B()), pick(A()))
print(stream(), stream(io.StringIO()))
