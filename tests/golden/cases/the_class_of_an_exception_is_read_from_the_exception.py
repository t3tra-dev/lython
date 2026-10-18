# WHAT: `type(e)`, `type(e).__name__` and `isinstance(e, t)` for an exception
# typed as a base -- a parameter, a `BaseException | None`, a value returned
# from a function -- answer the class the exception was raised as.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: a builtin exception typed
# `BaseException` may be any class of the taxonomy, which only its header
# says. The class was read from the STATIC type instead: `type(exc).__name__`
# for a ValueError held in a `BaseException | None` printed "BaseException",
# and `isinstance(e, ValueError)` through a class value was False for a
# ValueError passed to a function.
def check(e: BaseException, t: type[BaseException]) -> bool:
    return isinstance(e, t)


print(check(ValueError("v"), ValueError), check(ValueError("v"), LookupError))
print(check(KeyError("k"), LookupError), check(KeyError("k"), Exception))


def maybe(e: BaseException | None, t: type[BaseException]) -> bool:
    return isinstance(e, t)


print(maybe(ValueError("v"), ValueError), maybe(None, ValueError))


def name(e: BaseException | None) -> str:
    return type(e).__name__


print(name(ValueError("v")), name(KeyError("k")), name(None))


def made(n: int) -> BaseException:
    if n == 0:
        return KeyError("k")
    return ZeroDivisionError("z")


for n in [0, 1]:
    c = type(made(n))
    print(c.__name__, c is KeyError, issubclass(c, ArithmeticError), c)
