# probe: a module-level generator iterating an object's field (contrast with the method form)
# axes: op=generator-function flow=for
# CLASSIFICATION: 3 loud 拒否 (診断)
#   /Users/user/Desktop/dev/lython/.claude/worktrees/agent-a121ace38a4d2e6ff/tests/probe/known_module_generator_over_field.py:14:0: emit error: generator function return annotation is incompatible with inferred Generator or 
# CPython 3.14 expects: 6

from typing import Iterator


class Bag:
    def __init__(self, xs: list[int]) -> None:
        self.xs: list[int] = xs


def each(b: Bag) -> Iterator[int]:
    for x in b.xs:
        yield x


b = Bag([1, 2, 3])
total = 0
for v in each(b):
    total += v
print(total)
