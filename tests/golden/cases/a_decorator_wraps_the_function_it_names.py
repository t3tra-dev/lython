# WHAT: `@deco` on a module-level def and on a nested one, stacked, and with
# the decorated name recursing into itself.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: the wrapper is a function
# VALUE that has to reach the call and carry its capture with it. A decorator
# that resolved but lost `fn` would call the innermost function and print a
# plausible number; the stacked case pins the ORDER, which is the other thing
# a lost capture gets wrong.
#
# ⭐ AND A DECORATOR FACTORY, `@deco(arg)`, which is `f = deco(arg)(f)`. The
# refusal that stood here said the intermediate value is "a function the
# compiler would have to see through" -- while `shown = tag("v")(show)`, the
# hand-written spelling of the same three calls, compiled beside it. What was
# missing was the desugaring, and the factory's own call is simply the callee
# of the application.
#
# ⛔ THE DECORATED NAME IS A MODULE CELL, because that is what CPython makes
# it: every later reference -- a recursion inside the function's own body,
# another function calling it -- resolves the rebinding at CALL time and goes
# through the wrapper. A body binds the name to the emitted SYMBOL, which is
# the undecorated function, so a decorated `fib(6)` printed 9 for 33. Only the
# INNERMOST application reads the symbol; `@a @b def f` is two assignments and
# the second must read what the first stored.
def times_ten(fn):
    def wrapper(n: int) -> int:
        return fn(n) * 10
    return wrapper


def plus_one(fn):
    def wrapper(n: int) -> int:
        return fn(n) + 1
    return wrapper


@times_ten
def double(n: int) -> int:
    return n * 2


print(double(3))


@times_ten
@plus_one
def triple(n: int) -> int:
    return n * 3


print(triple(3))


@plus_one
@times_ten
def quad(n: int) -> int:
    return n * 4


print(quad(3))


@plus_one
def fib(n: int) -> int:
    if n < 2:
        return n
    return fib(n - 1) + fib(n - 2)


print(fib(6))


def call_through(n: int) -> int:
    return double(n)


print(call_through(3))


def outer() -> int:
    @times_ten
    def local(n: int) -> int:
        return n + 5
    return local(1)


print(outer())
print([double(v) for v in [1, 2]])


from typing import Callable


def tag(t: str) -> "Callable[[Callable[[int], str]], Callable[[int], str]]":
    def deco(fn: "Callable[[int], str]") -> "Callable[[int], str]":
        def wrapper(n: int) -> str:
            return t + fn(n)

        return wrapper

    return deco


def scaled(k: int) -> "Callable[[Callable[[int], int]], Callable[[int], int]]":
    def deco(fn: "Callable[[int], int]") -> "Callable[[int], int]":
        def wrapper(n: int) -> int:
            return fn(n) * k

        return wrapper

    return deco


def bump(fn: "Callable[[int], int]") -> "Callable[[int], int]":
    def wrapper(n: int) -> int:
        return fn(n) + 1

    return wrapper


@tag("a")
@tag("b")
def label(n: int) -> str:
    return str(n)


# A factory under a plain decorator: the order is what a lost capture gets
# wrong, and `bump` sees the SCALED function.
@bump
@scaled(3)
def go(n: int) -> int:
    return n


@scaled(2)
def fact(n: int) -> int:
    if n <= 1:
        return 1
    return n * fact(n - 1)


print(label(1), go(2), fact(4))
