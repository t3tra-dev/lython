# WHAT: generator methods that delegate to, iterate, and yield the result of a
# method of their own class written further down -- the shape difflib's
# Differ._fancy_replace has with _fancy_helper and _qformat.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: every one was refused
# ("does not provide manifest method 'later'") because a generator's yield
# type is read off its body before the methods below it are published; what
# has to hold is that each one yields what the later method gives it, in
# order, which only running them shows.
from collections.abc import Iterator


class Steps:
    def delegated(self, n: int) -> Iterator[str]:
        yield "start"
        yield from self.numbers(n)

    def iterated(self, n: int) -> Iterator[str]:
        for x in self.numbers(n):
            yield x + "!"

    def called(self, n: int) -> Iterator[str]:
        yield self.label(n)

    def numbers(self, n: int) -> Iterator[str]:
        for i in range(n):
            yield str(i)

    def label(self, n: int) -> str:
        return "label %d" % n


s = Steps()
print(list(s.delegated(3)), list(s.iterated(2)), list(s.called(7)))
