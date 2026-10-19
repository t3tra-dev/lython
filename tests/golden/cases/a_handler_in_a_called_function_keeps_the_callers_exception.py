# WHAT: an exception raised and handled inside a function that an `except`
# block calls -- or a `finally` block running for an exception -- leaves the
# outer exception the one being handled: the outer binding still reads it, a
# bare `raise` re-raises it, `sys.exc_info()` reports it, and the `finally`
# goes on to propagate it. A handler that completes by `return`, `break` or
# `continue` returns to it the same way.
#
# WHY THIS IS RUN: which exception is being handled after the call is decided
# by the run -- by which handler the inner exception was raised inside.
import sys


def handled() -> None:
    try:
        raise KeyError("inner")
    except KeyError:
        pass


def returns_from_its_handler() -> int:
    try:
        raise KeyError("inner")
    except KeyError:
        return 1


def leaves_its_handler_by_loop() -> int:
    seen = 0
    for i in range(3):
        try:
            raise IndexError("inner")
        except IndexError:
            seen += 1
            if i == 1:
                break
            continue
    return seen


def raises_through_a_finally() -> None:
    try:
        raise ValueError("in flight")
    finally:
        handled()
        try:
            raise KeyError("caught in finally")
        except KeyError:
            pass


try:
    raise ValueError("outer " + "x" * 200)
except ValueError as e:
    handled()
    print(len(str(e)), str(e)[:5])
    print(returns_from_its_handler(), leaves_its_handler_by_loop())
    print(type(sys.exc_info()[1]).__name__, len(str(sys.exc_info()[1])))
    try:
        raise
    except ValueError as again:
        print("re-raised", again is e)

try:
    raises_through_a_finally()
except ValueError as f:
    print("propagated", f)

for round in range(2):
    try:
        raise LookupError("round " + str(round))
    except LookupError as r:
        handled()
        print(r)
