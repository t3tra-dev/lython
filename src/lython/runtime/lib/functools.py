"""functools — higher-order functions, Lython port (static subset).

Port of CPython's Lib/functools.py, restricted to the well-typed statically
compilable surface. It ships as SOURCE inside the compiler: `import functools`
resolves this file through the same path as user source modules and compiles
it with the program. `reduce` is generic over the element type and
monomorphizes per instantiation.

CPython's `functools.reduce` is the `_functools` C accelerator, with the
Python body in Lib/functools.py as the fallback. This port keeps the Python
body as THE implementation: it compiles to native code here.

Deviations from CPython, pending language surface:
  - `reduce` takes a `list[T]`, not an arbitrary iterable, and the
    accumulator shares the element type (`Callable[[T, T], T]`). typeshed's
    other overload (`Callable[[S, T], S]` with a differently-typed initial)
    needs overload selection on an omitted argument
  - `reduce(f, seq, None)` written EXPLICITLY means "no initial value" here,
    where CPython would fold None as the initial value: the missing-argument
    sentinel is `None` rather than a private singleton, because a module-level
    sentinel object has no statically distinguishable type
  - not provided (each needs language surface outside this wave): `partial` /
    `partialmethod` (ParamSpec), `lru_cache` / `cache` / `wraps` /
    `singledispatch` / `total_ordering` / `cached_property` (general
    decorators, which are diagnosed rather than silently ignored),
    `cmp_to_key` (generic classes)
"""

from typing import Callable

__all__ = ["reduce"]


def reduce[T](function: Callable[[T, T], T], sequence: list[T],
              initial: T | None = None) -> T:
    """reduce(function, iterable[, initial]) -> value

    Apply a function of two arguments cumulatively to the items of a sequence
    or iterable, from left to right, so as to reduce the iterable to a single
    value.  For example, reduce(lambda x, y: x+y, [1, 2, 3, 4, 5]) calculates
    ((((1+2)+3)+4)+5).  If initial is present, it is placed before the items
    of the iterable in the calculation, and serves as a default when the
    iterable is empty.
    """
    if initial is None:
        if len(sequence) == 0:
            raise TypeError(
                "reduce() of empty iterable with no initial value")
        value = sequence[0]
        index = 1
    else:
        value = initial
        index = 0
    while index < len(sequence):
        value = function(value, sequence[index])
        index = index + 1
    return value
