# Why execution: the yielded sequence is the evidence the loop ran inside the
# frame. `for i in range(n): yield i` -- the most ordinary generator there is
# -- did not compile: "generator int yield lane has neither physical values
# nor primitive evidence", and so did a for-loop that merely PRECEDED a yield
# in the same body. A while-loop over the same counter worked, which is what
# made it look like a loop limit rather than what it was: the resume clone's
# split block argument has an (i64, valid) pair, and the evidence merge
# assigned the edge source's lane over it.
from typing import Iterator


def squares(n: int) -> Iterator[int]:
    for i in range(n):
        yield i * i


def stepped(n: int) -> Iterator[int]:
    for i in range(0, n, 2):
        yield i


def main() -> None:
    print(list(squares(4)))
    for v in squares(3):
        print(v)
    print(list(stepped(6)))
    print(sum(squares(5)))


main()
