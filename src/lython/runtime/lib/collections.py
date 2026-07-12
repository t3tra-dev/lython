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
  - update/subtract count elements of an iterable of keys
  - most_common(n) orders by count descending, insertion-ordered on ties
    (matches CPython's stable sort ordering)
  - total() sums the counts
Deviations from CPython, pending language surface:
  - Counter() takes no constructor arguments; seed with update(...)
  - most_common requires an explicit n (no None default)
  - elements(), arithmetic/comparison operators are not provided
"""

__all__ = ["Counter"]


class Counter:
    def __init__(self) -> None:
        self.data: dict[str, int] = {}

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

    def most_common(self, n: int) -> list[tuple[str, int]]:
        result: list[tuple[str, int]] = []
        taken: dict[str, int] = {}
        selected = 0
        while selected < n:
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
                selected = n
        return result
