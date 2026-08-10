# Why execution: the failure was a runtime abort, not a diagnostic. Building an
# object whose Optional field defaults to None released a header nobody had
# written -- `Ly_DecRef observed non-positive refcount` -- because the dead
# value standing in for "no previous field value" carried union tag 0, which
# names the FIRST member. For Optional[int] that is the int. Only running the
# constructor tells the two apart; the compiler exited 0 either way.
from typing import Optional


class WithInt:
    def __init__(self) -> None:
        self.value: Optional[int] = None


class WithStr:
    def __init__(self) -> None:
        self.name: Optional[str] = None


class WithList:
    def __init__(self) -> None:
        self.items: Optional[list[int]] = None


def main() -> None:
    print(WithInt().value is None)
    print(WithStr().name is None)
    print(WithList().items is None)


main()
