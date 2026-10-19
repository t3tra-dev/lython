# A call through a `Callable[[str], object]` value, in a program that also
# has a function taking `Iterable[str]` -- which a str is, so the dispatch
# keeps that function as a target the value might hold. Its arm passed the
# str as it was where the parameter is a box, and the whole program was
# refused ("cannot pass concrete object builtins.str as builtins.object").
# The str is boxed for that arm; a value whose handle could be mistaken for
# a box is still refused.
# Why run it: the dispatch has to reach the function the value holds, and
# the other arm has to stay compiled without running.
from collections.abc import Callable, Iterable


def joiner(parts: Iterable[str]) -> str:
    return "-".join(parts)


def shout(text: str) -> object:
    print(text.upper())
    return None


callbacks: list[Callable[[str], object]] = [shout, shout]
for callback in callbacks:
    callback("x")
j = joiner
print(j(["a", "b"]))
