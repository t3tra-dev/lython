# WHAT: a method whose `Iterable[...]` parameter defaults to `()`, called with
# the default and with lists, from a class built at module level and inside a
# function.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: the default was held in a
# cell typed by the protocol, and a protocol value has no reference count to
# take, so the lowering refused the read ("has no runtime retain primitive");
# the value each call sees is a run-time question.
from collections.abc import Iterable


class Suite:
    def __init__(self, tests: Iterable[int] = ()) -> None:
        self.tests: list[int] = []
        for t in tests:
            self.tests.append(t)


def build() -> list[int]:
    return Suite([4, 5]).tests + Suite().tests


print(Suite().tests, Suite([1, 2]).tests, build())
