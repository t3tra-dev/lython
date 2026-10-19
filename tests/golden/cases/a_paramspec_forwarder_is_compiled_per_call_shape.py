# WHAT: a function whose `*args: P.args` and `**kwargs: P.kwargs` forward to
# `f: Callable[P, R]` passes each call's arguments on with the arity and the
# names that call gave -- with no other type parameter, after a positional-
# only `f` (so a keyword spelled `f` reaches **kwargs), after a defaulted
# parameter, from a forwarder in another module or another forwarder, from a
# generator, a recursive call, a `try`, and a lambda kept for later; with
# `Concatenate`; and the packs read as a tuple and a dict hold what CPython's
# do.
#
# WHY THIS IS RUN: which function runs and what it returns are run-time
# values, and so are the tuple and the dict the body prints.
import typing
from collections.abc import Callable, Iterator
from typing import Concatenate

import a_paramspec_forwarder_is_compiled_per_call_shape_lib as lib
from a_paramspec_forwarder_is_compiled_per_call_shape_lib import call


def show(a: int, b: str = "x", *, sep: str = "-") -> None:
    print(a, b, sep)


def add(a: int, b: int = 0) -> int:
    return a + b


def run[**P](f: Callable[P, None], *args: P.args, **kwargs: P.kwargs) -> None:
    print("run", len(args), sorted(kwargs))
    f(*args, **kwargs)


run(show, 1)
run(show, 2, "y")
run(show, 3, sep="+")
run(show, 4, b="z", sep="/")
run(show, a=5)


def call_f[**P, R](f: Callable[P, R], /, *args: P.args,
                   **kwargs: P.kwargs) -> R:
    return f(*args, **kwargs)


def takes_f(f: int) -> int:
    return f * 10


print(call_f(takes_f, f=3), call_f(takes_f, 4))


def tries[**P, R](f: Callable[P, R], times: int = 1, *args: P.args,
                  **kwargs: P.kwargs) -> list[R]:
    return [f(*args, **kwargs) for _ in range(times)]


print(tries(add, 2, 5), tries(add, 3, 5, b=10), tries(add, 1, 7, 8))
print(call(add, 1, 2), lib.twice(add, 3, b=4))


class Counter:
    def __init__(self, start: int) -> None:
        self.n = start

    def bump(self, by: int = 1) -> int:
        self.n += by
        return self.n


c = Counter(10)
print(call(c.bump), call(c.bump, 5), call(c.bump, by=100))


def gen[**P](f: Callable[P, int], *args: P.args,
             **kwargs: P.kwargs) -> Iterator[int]:
    yield f(*args, **kwargs)
    yield f(*args, **kwargs) * 2


print(list(gen(add, 7)), list(gen(add, 7, b=1)))


def countdown[**P](f: Callable[P, None], n: int, *args: P.args,
                   **kwargs: P.kwargs) -> None:
    if n <= 0:
        return
    f(*args, **kwargs)
    countdown(f, n - 1, *args, **kwargs)


countdown(show, 2, 6)
countdown(show, 2, 7, sep="!")


def raises[**P](exc: type[BaseException], f: Callable[P, object],
                *args: P.args, **kwargs: P.kwargs) -> bool:
    try:
        f(*args, **kwargs)
    except exc:
        return True
    return False


def div(a: int, b: int) -> float:
    return a / b


print(raises(ZeroDivisionError, div, 1, 0), raises(ZeroDivisionError, div, 1, b=2))


def defer[**P](f: Callable[P, None], *args: P.args,
               **kwargs: P.kwargs) -> Callable[[], None]:
    return lambda: f(*args, **kwargs)


later = [defer(show, 8), defer(show, 9, "w", sep="|")]
for thunk in later:
    thunk()


def describe[**P, R](f: Callable[P, R], *args: P.args,
                     **kwargs: P.kwargs) -> str:
    return repr(args) + " " + repr(kwargs) + " -> " + repr(f(*args, **kwargs))


print(describe(add, 1))
print(describe(add, 1, b=2))
print(describe(show, 1, "q", sep="!"))


def with_prefix[**P, R](f: Callable[Concatenate[str, P], R], *args: P.args,
                        **kwargs: P.kwargs) -> R:
    return f("pre", *args, **kwargs)


def with_tag[**P, R](f: Callable[typing.Concatenate[str, P], R],
                     *args: P.args, **kwargs: P.kwargs) -> R:
    return f("tag", *args, **kwargs)


def join(head: str, n: int, *, sep: str = "-") -> str:
    return head + sep + str(n)


print(with_prefix(join, 3), with_prefix(join, 4, sep="+"), with_tag(join, 5))
