# WHAT: `@deco` on a module-level def and on a nested one, stacked, and with
# the decorated name recursing into itself.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: the wrapper is a function
# VALUE that has to reach the call and carry its capture with it. A decorator
# that resolved but lost `fn` would call the innermost function and print a
# plausible number; the stacked case pins the ORDER, which is the other thing
# a lost capture gets wrong.
#
# ⛔ A decorator FACTORY (`@deco(arg)`) is still refused: it is one more call
# whose intermediate value is a function, and a partial answer there would be
# a wrong wrapper rather than a diagnostic.
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
