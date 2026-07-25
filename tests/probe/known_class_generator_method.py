# probe: REPORTED loud: a generator method on a class
# axes: op=generator-method flow=for
# CLASSIFICATION: 3 loud 拒否 (診断)
#   /Users/user/Desktop/dev/lython/.claude/worktrees/agent-a121ace38a4d2e6ff/tests/probe/known_class_generator_method.py:13:4: emit error: generator function return annotation is incompatible with inferred Generator or Async
# CPython 3.14 expects: 6

from typing import Iterator


class Bag:
    def __init__(self, xs: list[int]) -> None:
        self.xs: list[int] = xs

    def each(self) -> Iterator[int]:
        for x in self.xs:
            yield x


b = Bag([1, 2, 3])
total = 0
for v in b.each():
    total += v
print(total)
