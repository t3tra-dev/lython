# `f is g` and `f == g` over two `Callable` locals are refused:
#
#     v: Callable[[int], int] = double
#     w: Callable[[int], int] = double
#     print(v is w)     # `is` requires reference-typed operands ...
#     print(v == w)     # ... does not provide manifest method '__eq__'
#
# MEASURED (2026-09-06, RelWithDebInfo, today's tree). The refusal LOOKS like
# the four storability gates that excluded a function object for the same
# reason -- its static type is spelled `py.callable` rather than `py.contract`
# -- and all four of those were repairs. This one is not.
#
# ⛔ LETTING IT THROUGH IS A SILENT WRONG ANSWER, and that is the measurement:
# with `CallableType` accepted as reference-typed, `v is w` for two references
# to the SAME `def` compiles and prints False where CPython prints True. A
# function VALUE is materialized at each reference here, so two reads of one
# `def` build two objects with two addresses.
#
# ⭐ WHAT IT WOULD TAKE: one function object per def, cached at the binding, so
# every reference hands back the same handle. That is also what would make
# `==` work, since CPython's `==` on functions IS identity.
#
# ⛔ AND THE SHAPES THAT DO WORK, which is what keeps this narrow: an
# `Optional[Callable]` compared against None (`v is None`) is correct -- the
# tag decides it and no address is read -- and so is calling either operand.
from typing import Callable, Optional


def double(n: int) -> int:
    return n * 2


def use() -> int:
    v: Callable[[int], int] = double
    w: Callable[[int], int] = double
    maybe: Optional[Callable[[int], int]] = double
    return (1 if v is w else 0) + (0 if maybe is None else v(2))


print(use())
