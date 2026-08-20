# What this pins: a wrapper handed a function that itself carries captures
# RAISES instead of answering.
#
#     f = double(base)      # each returns a wrapper that calls its captured fn
#     f = add_one(f)
#     print(f(5))           # CPython prints 11
#
# This printed 6 -- base(5) + 1 -- for one commit. The wrapper's `fn(n)` was
# devirtualized to a direct call to `base`, because the indirect-call lowering
# took the single surviving candidate as the answer and a runtime function
# value, carrying no closure evidence, leaves only the zero-closure candidates
# standing. The fast path now needs a statically known target, so an unknown one
# goes through the id dispatch and its default arm raises.
#
# Why this must run: the whole point is WHICH failure happens. A compile-time
# refusal would be a different (and better) answer, but the compiler cannot
# prove the value is a closure-carrying function -- so the check is the runtime
# one, and only running it shows that it fires.
#
# ⛔ A closure is compile-time evidence here: LyFunction_New writes five zero
# words where the captures would go. Until a function object carries them, this
# is the honest answer, and `double(base)(5)` -- one level -- still works.
from typing import Callable


def add_one(fn: Callable[[int], int]) -> Callable[[int], int]:
    def w(n: int) -> int:
        return fn(n) + 1

    return w


def double(fn: Callable[[int], int]) -> Callable[[int], int]:
    def w(n: int) -> int:
        return fn(n) * 2

    return w


def base(n: int) -> int:
    return n


print(double(base)(5))
first = double(base)
second = add_one(first)
print(second(5))
