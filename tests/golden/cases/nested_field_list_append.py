# Why execution: the second append silently corrupted the list. The compiler
# exited 0 and the program printed a garbage integer, aborted with
# `Ly_IncRef observed non-positive refcount`, or segfaulted -- different
# answers from one binary. Only running tells those from the right value.
#
# The cause was one level up from the mutation: `t.mid.leaves.append` publishes
# the grown list into `mid`'s slot, but `t`'s own cached description of `mid`
# still held the pre-append `leaves`, so the second `t.mid` read it and grew a
# list described by the first append's payload and length.


class Leaf:
    def __init__(self, n: int) -> None:
        self.n: int = n


class Mid:
    def __init__(self) -> None:
        self.numbers: list[int] = []
        self.leaves: list[Leaf] = []


class Top:
    def __init__(self) -> None:
        self.mid: Mid = Mid()


def main() -> None:
    top = Top()
    top.mid.numbers.append(1)
    top.mid.numbers.append(2)
    top.mid.numbers.append(3)
    print(len(top.mid.numbers))
    print(top.mid.numbers[1])
    print(top.mid.numbers[2])

    top.mid.leaves.append(Leaf(7))
    top.mid.leaves.append(Leaf(8))
    print(top.mid.leaves[1].n)


main()
