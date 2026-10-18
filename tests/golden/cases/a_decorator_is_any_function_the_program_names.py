# WHAT: a decorator imported from another module rebinds the function it
# decorates -- by its name, through the module (`decorators.twice`), as a
# factory call, stacked over another, and on a def nested in a function --
# and the recursion inside a decorated function reads the rebound name.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: the answers are the
# wrappers' own, and the calls the logging decorator records are made at run
# time. Every one was refused ("decorator 'twice' is not supported"): only a
# def of the same module could decorate.
import a_module_of_decorators
from a_module_of_decorators import logged, times, twice


@twice
def inc(n: int) -> int:
    return n + 1


@times(3)
def dec(n: int) -> int:
    return n - 1


@a_module_of_decorators.twice
def dbl(n: int) -> int:
    return n * 2


@logged
@twice
def sq(n: int) -> int:
    return n * n


@logged
def countdown(n: int) -> int:
    return 0 if n == 0 else countdown(n - 1) + 1


def outer(k: int) -> int:
    @times(k)
    def inner(n: int) -> int:
        return n + k
    return inner(1)


print(inc(1), dec(5), dbl(3), sq(2), countdown(2), outer(4))
print(a_module_of_decorators.calls)
