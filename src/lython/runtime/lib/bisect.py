"""bisect — array bisection algorithms, Lython port.

Port of CPython's Lib/bisect.py, restricted to the well-typed statically
compilable surface. It ships as SOURCE inside the compiler: `import bisect`
resolves this file through the same path as user source modules and compiles
it with the program. The functions are generic over the element type and
monomorphize per instantiation, so `bisect_right(list[int], int)` and
`bisect_right(list[str], str)` compile to separate specializations.

Signatures follow typeshed stdlib/bisect.pyi (specialized to `list`, see
below); the bodies are CPython's, including the deliberate use of `<` only,
which matches the `__lt__` logic of list.sort() and heapq.

CPython replaces these Python bodies with the `_bisect` C accelerator at
import time (`from _bisect import *`). This port keeps the Python bodies as
THE implementation: they compile to native code here, so a separate native
module would buy nothing.

Deviations from CPython, pending language surface:
  - the sequence is `list[T]`, not an arbitrary `MutableSequence[T]` /
    `Sequence[T]` (typeshed's parameter type) — protocol-typed sequence
    receivers carry no concrete method evidence for the subscript
  - no `key=` argument: it is keyword-only and Optional-callable, and
    calling through an Optional callable parameter is not statically
    resolvable yet. Pass a decorated sequence instead
  - insort_left / insort_right are declared with CPython's semantics but
    cannot be CALLED yet: inserting into a caller-owned list through a
    parameter needs borrowed-container mutation, which the ownership layer
    rejects. A call site gets an explicit diagnostic (never a silent
    mis-insert); bisect_left / bisect_right are unaffected
"""

__all__ = ["bisect_left", "bisect_right", "bisect", "insort_left",
           "insort_right", "insort"]


def bisect_right[T](a: list[T], x: T, lo: int = 0, hi: int | None = None) -> int:
    """Return the index where to insert item x in list a, assuming a is sorted.

    The return value i is such that all e in a[:i] have e <= x, and all e in
    a[i:] have e > x.  So if x already appears in the list, a.insert(i, x) will
    insert just after the rightmost x already there.

    Optional args lo (default 0) and hi (default len(a)) bound the
    slice of a to be searched.
    """
    if lo < 0:
        raise ValueError("lo must be non-negative")
    # Fresh copies, not the parameters themselves: a borrowed entry argument
    # that is loop-carried and returned trips the ownership verifier.
    low = lo + 0
    if hi is None:
        high = len(a)
    else:
        high = hi + 0
    while low < high:
        mid = (low + high) // 2
        if x < a[mid]:
            high = mid
        else:
            low = mid + 1
    return low


def bisect_left[T](a: list[T], x: T, lo: int = 0, hi: int | None = None) -> int:
    """Return the index where to insert item x in list a, assuming a is sorted.

    The return value i is such that all e in a[:i] have e < x, and all e in
    a[i:] have e >= x.  So if x already appears in the list, a.insert(i, x)
    will insert just before the leftmost x already there.

    Optional args lo (default 0) and hi (default len(a)) bound the
    slice of a to be searched.
    """
    if lo < 0:
        raise ValueError("lo must be non-negative")
    low = lo + 0
    if hi is None:
        high = len(a)
    else:
        high = hi + 0
    while low < high:
        mid = (low + high) // 2
        if a[mid] < x:
            low = mid + 1
        else:
            high = mid
    return low


def insort_right[T](a: list[T], x: T, lo: int = 0,
                    hi: int | None = None) -> None:
    """Insert item x in list a, and keep it sorted assuming a is sorted.

    If x is already in a, insert it to the right of the rightmost x.

    Optional args lo (default 0) and hi (default len(a)) bound the
    slice of a to be searched.
    """
    index = bisect_right(a, x, lo, hi)
    a.insert(index, x)


def insort_left[T](a: list[T], x: T, lo: int = 0,
                   hi: int | None = None) -> None:
    """Insert item x in list a, and keep it sorted assuming a is sorted.

    If x is already in a, insert it to the left of the leftmost x.

    Optional args lo (default 0) and hi (default len(a)) bound the
    slice of a to be searched.
    """
    index = bisect_left(a, x, lo, hi)
    a.insert(index, x)


# Create aliases (CPython does the same, after the _bisect import).
bisect = bisect_right
insort = insort_right
