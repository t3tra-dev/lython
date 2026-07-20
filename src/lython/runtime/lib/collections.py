"""collections — container datatypes, Lython port.

This is Lython's port of CPython's Lib/collections/__init__.py, restricted
to the well-typed statically compilable surface. It ships as SOURCE inside
the compiler: `import collections` resolves this file through the same path
as user source modules and compiles it with the program.

Counter follows typeshed stdlib/collections/__init__.pyi specialized to
str keys (the runtime dict's key surface). CPython's Counter subclasses
dict; dict subclassing is outside the static surface, so this port is a
COMPOSITION over a `dict[str, int]` field with the same observable method
semantics:
  - c[missing] returns 0 (CPython Counter.__missing__), and does NOT insert
  - Counter(iterable) counts elements of an iterable of keys, like update
  - update/subtract count elements of an iterable of keys
  - most_common() orders by count descending, insertion-ordered on ties
    (matches CPython's stable sort ordering); most_common(None) == all
  - +, -, |, & implement the multiset operations with CPython's result
    ordering (self's keys in insertion order, then other's novel keys)
  - == compares counts with missing keys treated as zero (CPython 3.10+)
  - total() sums the counts
Deviations from CPython, pending language surface:
  - keys are str only (typeshed's Counter is generic)
  - Counter()/update()/subtract() seed from a list of keys; the
    mapping/kwargs constructor forms are not provided
  - elements() returns a materialized list, not a lazy iterator
  - <, <=, >, >= multiset comparisons and unary +/- are not provided
"""

__all__ = ["Counter"]


class Counter:
    def __init__(self, iterable: list[str] | None = None) -> None:
        self.data: dict[str, int] = {}
        if iterable is not None:
            self.update(iterable)

    def __getitem__(self, key: str) -> int:
        if key in self.data:
            return self.data[key]
        return 0

    def __setitem__(self, key: str, count: int) -> None:
        self.data[key] = count

    def __contains__(self, key: str) -> bool:
        return key in self.data

    def __len__(self) -> int:
        return len(self.data)

    def __eq__(self, other: "Counter") -> bool:
        for key in self.data:
            if self.data[key] != other[key]:
                return False
        for key in other.data:
            if key not in self.data:
                if other.data[key] != 0:
                    return False
        return True

    def __add__(self, other: "Counter") -> "Counter":
        result = Counter()
        for key in self.data:
            newcount = self.data[key] + other[key]
            if newcount > 0:
                result.data[key] = newcount
        for key in other.data:
            if key not in self.data:
                if other.data[key] > 0:
                    result.data[key] = other.data[key]
        return result

    def __sub__(self, other: "Counter") -> "Counter":
        result = Counter()
        for key in self.data:
            newcount = self.data[key] - other[key]
            if newcount > 0:
                result.data[key] = newcount
        for key in other.data:
            if key not in self.data:
                if other.data[key] < 0:
                    result.data[key] = 0 - other.data[key]
        return result

    def __or__(self, other: "Counter") -> "Counter":
        # max(self[k], other[k]) over positive results, spelled through the
        # multiset identity  self | other == self + (other - self)  because
        # the direct formulation trips a release-placement compiler bug
        # (conditional dict writes after two owned dict reads in a loop);
        # reported to the Wave 2 foundations track. The identity preserves
        # CPython's result ordering (self's keys, then other's novel keys).
        return self + (other - self)

    def __and__(self, other: "Counter") -> "Counter":
        # min(self[k], other[k]) over positive results; same identity trick:
        # self & other == self - (self - other). Iterates self's keys only,
        # matching CPython's ordering.
        return self - (self - other)

    def update(self, iterable: list[str]) -> None:
        for elem in iterable:
            if elem in self.data:
                self.data[elem] = self.data[elem] + 1
            else:
                self.data[elem] = 1

    def subtract(self, iterable: list[str]) -> None:
        for elem in iterable:
            if elem in self.data:
                self.data[elem] = self.data[elem] - 1
            else:
                self.data[elem] = -1

    def total(self) -> int:
        result = 0
        for key in self.data:
            result = result + self.data[key]
        return result

    def elements(self) -> list[str]:
        # Why concat-rebind instead of an inner append loop: a nested
        # append loop over the dict-iteration key blows past the
        # affine-ownership verifier's state budget (reported to the
        # Wave 2 foundations track); repetition + concat stays linear.
        result: list[str] = []
        for key in self.data:
            result = result + [key] * self.data[key]
        return result

    def most_common(self, n: int | None = None) -> list[tuple[str, int]]:
        limit = len(self.data)
        if n is not None:
            if n < limit:
                limit = n
        result: list[tuple[str, int]] = []
        taken: dict[str, int] = {}
        selected = 0
        while selected < limit:
            best_key = ""
            best_count = 0
            found = False
            for key in self.data:
                if key not in taken:
                    value = self.data[key]
                    if not found:
                        best_key = key
                        best_count = value
                        found = True
                    elif value > best_count:
                        best_key = key
                        best_count = value
            if found:
                result.append((best_key, best_count))
                taken[best_key] = 1
                selected = selected + 1
            else:
                selected = limit
        return result
