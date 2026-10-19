# A generator hands a factory's result to a @contextmanager METHOD and
# suspends inside its `with` -- the shape of unittest's subTest. The value is
# kept for the frame (its class can hold any object, so its release is
# observable) and enters the `with` body through an edge that lends it a
# reference of its own; a later cleanup merged that edge away, the lent
# reference then named the outer value, and the verifier read the outer
# release as the death of what the suspension still returned.
# Why run it: the inner context manager must see the object it was given
# after the suspension, and exit exactly once.
import contextlib
from collections.abc import Iterator


class Sub:
    params: dict[str, object]

    def __init__(self, tag: str) -> None:
        self.tag = tag


class Outcome:
    def __init__(self) -> None:
        self.parts: list[str] = []

    @contextlib.contextmanager
    def part(self, sub: Sub) -> Iterator[None]:
        try:
            yield
        finally:
            self.parts.append(sub.tag)


def make(tag: str) -> Sub:
    return Sub(tag)


class Case:
    def __init__(self) -> None:
        self.outcome = Outcome()

    @contextlib.contextmanager
    def sub_test(self, tag: str) -> Iterator[None]:
        sub = make(tag)
        with self.outcome.part(sub):
            yield


case = Case()
for tag in ["a", "b"]:
    with case.sub_test(tag):
        print("inside", tag)
print(case.outcome.parts)
