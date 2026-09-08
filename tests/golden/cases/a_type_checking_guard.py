# Why execution: `TYPE_CHECKING` is False at RUN TIME and true for the checker,
# so the only way to say the compiler got both halves is to run the program:
# the guarded block must not execute, its imports must still be bound, and the
# `else` arm must be the one that runs.
#
# The name had no binding at all -- "unresolved name 'TYPE_CHECKING'" -- so the
# two lines an annotated module opens with took the whole program down.
#
# ⭐ It binds to the LITERAL False, not to `bool`: an import inside the block
# is a declaration for the CHECKER, and this compiler's checker is its type
# system, so the annotation below has to resolve while the print above it must
# not run.
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Callable
    from typing import Optional

    RAN_CHECK = "yes"

import typing


def apply(fn: "Callable[[int], int]", n: int) -> int:
    return fn(n)


def first(v: "Optional[int]") -> int:
    return 0 if v is None else v


def main() -> None:
    print(TYPE_CHECKING, typing.TYPE_CHECKING)
    if TYPE_CHECKING:
        print("this line never runs")
    if not TYPE_CHECKING:
        print("this one does")
    print(apply(lambda n: n * 2, 21))
    print(first(3), first(None))


main()
