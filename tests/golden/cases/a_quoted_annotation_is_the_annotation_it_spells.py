# A quoted annotation resolved only when it was a bare name: `"list[Key]"`,
# `"tuple[Node, int]"` and `"Callable[[int], int]"` all became `object`, and the
# program then failed as "static type builtins.object does not provide ...",
# which names neither the annotation nor the quoting. A method returning its own
# class writes the second form, and `from __future__ import annotations` makes
# every annotation a string. Must run: the values are what says the element type
# survived the quoting -- an `object` element would have refused, but a wrongly
# resolved one would not.


class Key:
    def __init__(self, v: int) -> None:
        self.v = v

    def twin(self) -> "Key":
        return Key(self.v)

    def pair(self) -> "tuple[Key, int]":
        return (self.twin(), self.v)


keys: "list[Key]" = [Key(1), Key(2)]
total = 0
for k in keys:
    total = total + k.v
print(total, len(keys))

k = Key(3)
print(k.twin().v, k.pair()[1])


def count(xs: "list[int]") -> "int":
    return len(xs)


def joined(p: "tuple[int, str]") -> str:
    return str(p[0]) + p[1]


def nested(d: "dict[str, list[int]]") -> int:
    t = 0
    for name in sorted(d):
        for v in d[name]:
            t = t + v
    return t


print(count([1, 2, 3]), joined((1, "a")), nested({"a": [1, 2], "b": [3]}))

from typing import Callable


def apply(f: "Callable[[int], int]", v: int) -> int:
    return f(v)


print(apply(lambda n: n * 3, 4))

# The unquoted spellings, side by side, so a divergence between them shows.
def count_plain(xs: list[int]) -> int:
    return len(xs)


print(count_plain([1, 2, 3]) == count([1, 2, 3]))
