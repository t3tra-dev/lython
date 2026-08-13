# Returning a generator out of a function and iterating it at the call site.
# The refusal used to be "runtime manifest has no types.GeneratorType.__next__
# method" -- a sentence about the manifest for a program that did nothing to
# it -- while the same generator bound to a local iterates fine, which is the
# shape the message now points at.
from typing import Iterator


def inner() -> Iterator[int]:
    yield 1
    yield 2


def outer() -> Iterator[int]:
    return inner()


for v in outer():
    print(v)
