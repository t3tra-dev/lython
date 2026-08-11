# Why execution: these did not compile -- "operation with block successors
# must terminate its parent block" -- so what the golden pins is that each
# form now produces CPython's exact line, including the separator placement
# that a mis-ordered join would get wrong.
#
# A reducer or a comprehension lowers to a loop, which splits the block it is
# emitted into. The multi-argument print path saved an insertion point before
# emitting an argument speculatively and rewound to it afterwards; once the
# block had been split, that point named a block already ending in a
# terminator. Each of these printed correctly as the only argument, which is
# what made it read as a multi-argument problem.


class Boxed:
    def __init__(self) -> None:
        self.width: int = 3

    @property
    def area(self) -> int:
        return self.width


def main() -> None:
    numbers: list[int] = [3, 1, 4]
    print(sum(numbers), 1)
    print(min(numbers), 1)
    print(max(numbers), "end")
    print(sum(numbers), max(numbers))

    print("a", [x for x in [1, 2]])
    print([x for x in [1]], "m", [y for y in [2]])

    print(Boxed().area, 1)

    print(max("a", "b"), 1)
    print("a", 1, "b")


main()
