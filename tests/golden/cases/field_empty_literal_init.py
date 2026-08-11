# Why execution: these did not compile. An empty literal has nothing to infer
# an element type from, so `self.xs = []` produced list[object] and the store
# into a declared list[int] field was refused. What the golden pins is the
# VALUE, because a fix that guessed the wrong element type would compile and
# then mis-store.
#
# Writing the annotation inline already worked -- AnnAssign passes it down as
# the expected type. This is the same expectation, read from the field the
# target names.


class Accumulator:
    numbers: list[int]
    names: list[str]
    index: dict[str, int]

    def __init__(self) -> None:
        self.numbers = []
        self.names = []
        self.index = {}


class Inline:
    def __init__(self) -> None:
        self.numbers: list[int] = []


class Seeded:
    numbers: list[int]

    def __init__(self) -> None:
        self.numbers = [1]


def main() -> None:
    acc = Accumulator()
    print(acc.numbers)
    print(acc.names)
    print(len(acc.index))

    acc.numbers.append(1)
    acc.numbers.append(2)
    print(acc.numbers)

    acc.index["k"] = 7
    print(acc.index["k"])

    print(Inline().numbers)
    print(Seeded().numbers)


main()
