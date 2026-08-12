# Why execution: the elements are the evidence that the class's own __iter__
# ran. `for v in Box(...)` died in the lowering as "runtime manifest has no
# Box.__iter__ method" -- py.iter resolves its target against the manifest --
# while __len__, __getitem__ and __contains__ on the same class all worked,
# and so did `async for` over a class's __aiter__, which is this same shape in
# the async loop below the sync one.
from typing import Iterator


class Box:
    def __init__(self, items: list[int]) -> None:
        self.items = items

    def __iter__(self) -> Iterator[int]:
        return iter(self.items)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> int:
        return self.items[index]

    def __contains__(self, value: int) -> bool:
        return value in self.items


def main() -> None:
    b = Box([1, 2, 3])
    for v in b:
        print(v)
    print([v for v in b], list(b), sum(v for v in b))
    print(len(b), b[0], 2 in b, 9 in b)
    for pair in zip(b, b):
        print(pair)
    print(sorted(Box([3, 1, 2])))


main()
