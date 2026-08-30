# What: `in` over a class with no `__contains__`. CPython iterates, so the
# answer depends on the elements -- running it is what shows the walk happened
# and stopped at the right one. Both iteration shapes are here: `__iter__`, and
# the `__len__`/`__getitem__` sequence pair.
from typing import Iterator


class Bag:
    def __init__(self, items: "list[str]") -> None:
        self.items = items

    def __iter__(self) -> "Iterator[str]":
        return iter(self.items)


class Squares:
    def __init__(self, count: int) -> None:
        self.count = count

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int) -> int:
        if index >= self.count:
            raise IndexError("out of range")
        return index * index


bag = Bag(["red", "green"])
print("red" in bag, "blue" in bag)
print("red" not in bag, "blue" not in bag)

squares = Squares(4)
print(4 in squares, 5 in squares, 9 in squares)
print(list(squares), len(squares))
print(1 in [1, 2], "a" in "abc", 3 in {3: "x"})
