# Why execution: these did not compile at all -- the ownership walk's visited
# key grew by one every iteration, so its fixpoint never closed and the
# 20000-state cap fired. What the golden pins is the VALUE each accumulator
# ends with, since a walk that converges on the wrong reading would still
# compile.


def minimum_over_a_literal() -> None:
    lo: int = 0
    for value in [4, -2, 9]:
        if value < lo:
            lo = value
    print(lo)


def maximum_over_a_literal() -> None:
    hi: int = 0
    for value in [1, 7, 3]:
        if value > hi:
            hi = value
    print(hi)


def best_string() -> None:
    best: str = ""
    for text in ["b", "a", "c"]:
        if text > best:
            best = text
    print(best)


def counted() -> None:
    negatives: int = 0
    for value in [1, -2, -3, 4]:
        if value < 0:
            negatives = negatives + 1
    print(negatives)


def main() -> None:
    minimum_over_a_literal()
    maximum_over_a_literal()
    best_string()
    counted()


main()
