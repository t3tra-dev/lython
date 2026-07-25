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

OrderedDict follows typeshed stdlib/collections/__init__.pyi and CPython's
PURE-PYTHON OrderedDict in Lib/collections/__init__.py (the one _collections
accelerates), so this file is its CPython layer. It is generic in its key and
value types (`OrderedDict[K, V]`). CPython's subclasses dict; dict subclassing
is outside the static surface, so this port is a COMPOSITION over a
`dict[K, V]` field, which needs no linked-list bookkeeping at all: Lython's
dict is insertion-ordered (compact dict), so the field IS the order, and
move_to_end/popitem are delete+reinsert and first/last key reads.
  - repr is CPython 3.12+'s `OrderedDict({'a': 1})` / `OrderedDict()`
  - == against another OrderedDict is ORDER-SENSITIVE (CPython)
  - popitem(last=True) pops the newest, popitem(last=False) the oldest
  - move_to_end(key, last=True) moves to either end, KeyError if absent
Deviations from CPython, pending language surface:
  - keys()/values()/items() return materialized lists, not lazy views (no
    generator methods on classes yet); there is no __iter__, so `for k in od`
    is spelled `for k in od.keys()`
  - the constructor takes no iterable/mapping/kwargs seed; build with
    __setitem__ or update()
  - reversed(), __ior__/__or__, fromkeys() and __reduce__ are not provided
  - == against a plain dict (CPython's order-insensitive arm) is not provided

deque and defaultdict are NOT here: CPython implements both only in C
(Modules/_collectionsmodule.c, no pure-Python fallback in Lib), so per the
Lib/Modules layering rule they belong in runtime/modules/, not in this file.
"""

__all__ = ["Counter", "OrderedDict"]


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


class OrderedDict[K, V]:
    def __init__(self) -> None:
        self.data: dict[K, V] = {}

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, key: K) -> V:
        return self.data[key]

    def __setitem__(self, key: K, value: V) -> None:
        self.data[key] = value

    def __delitem__(self, key: K) -> None:
        del self.data[key]

    def __contains__(self, key: K) -> bool:
        return key in self.data

    def __eq__(self, other: OrderedDict[K, V]) -> bool:
        # Order-sensitive, unlike dict.__eq__: CPython compares two
        # OrderedDicts as ordered sequences of items.
        if len(self.data) != len(other.data):
            return False
        mine = self.keys()
        theirs = other.keys()
        index = 0
        while index < len(mine):
            if mine[index] != theirs[index]:
                return False
            if self.data[mine[index]] != other.data[theirs[index]]:
                return False
            index = index + 1
        return True

    def __repr__(self) -> str:
        if len(self.data) == 0:
            return "OrderedDict()"
        parts: list[str] = []
        for key in self.data:
            # Built by rebinding one local instead of a single concat
            # expression: two owned repr() results live across each other's
            # may-unwind call, which the ownership landing pad does not
            # release yet (reported to the Wave 3 foundations track).
            entry = repr(key)
            entry = entry + ": "
            entry = entry + repr(self.data[key])
            parts.append(entry)
        return "OrderedDict({" + ", ".join(parts) + "})"

    def keys(self) -> list[K]:
        result: list[K] = []
        for key in self.data:
            result.append(key)
        return result

    def values(self) -> list[V]:
        result: list[V] = []
        for key in self.data:
            result.append(self.data[key])
        return result

    def items(self) -> list[tuple[K, V]]:
        result: list[tuple[K, V]] = []
        for key in self.data:
            result.append((key, self.data[key]))
        return result

    def get(self, key: K, default: V) -> V:
        if key in self.data:
            return self.data[key]
        return default

    def setdefault(self, key: K, default: V) -> V:
        if key in self.data:
            return self.data[key]
        self.data[key] = default
        return default

    def pop(self, key: K) -> V:
        value = self.data[key]
        del self.data[key]
        return value

    def popitem(self, last: bool = True) -> tuple[K, V]:
        keys = self.keys()
        if len(keys) == 0:
            raise KeyError("dictionary is empty")
        if last:
            key = keys[len(keys) - 1]
        else:
            key = keys[0]
        value = self.data[key]
        del self.data[key]
        return (key, value)

    def move_to_end(self, key: K, last: bool = True) -> None:
        # A missing key raises through the backing dict's own lookup, which
        # already carries the key CPython-style — exactly what CPython does
        # (its move_to_end indexes self.__map first).
        value = self.data[key]
        del self.data[key]
        if last:
            # Reinsertion after deletion IS the move: the backing dict is
            # insertion-ordered, so the reinserted key lands last.
            self.data[key] = value
            return
        # Moving to the FRONT has no delete-and-reinsert spelling, so the
        # remaining items are re-appended behind the moved key.
        rest = self.keys()
        restvalues = self.values()
        self.data.clear()
        self.data[key] = value
        index = 0
        while index < len(rest):
            self.data[rest[index]] = restvalues[index]
            index = index + 1

    def clear(self) -> None:
        self.data.clear()

    def copy(self) -> OrderedDict[K, V]:
        result = OrderedDict[K, V]()
        for key in self.data:
            result.data[key] = self.data[key]
        return result

    def update(self, other: OrderedDict[K, V]) -> None:
        for key in other.data:
            self.data[key] = other.data[key]
