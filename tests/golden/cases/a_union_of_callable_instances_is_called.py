# WHAT: a list holding instances of two classes that each declare
# `__call__`, as a test suite holds cases and nested suites, called with an
# argument through the union element type; and one held in a name.
#
# WHY THIS IS RUN: which `__call__` runs is the element's class at run time.
# Calling a union value was refused ("no manifest __call__ contract"),
# although calling its `__call__` method by name was answered member by
# member.
class Case:
    def __init__(self, name: str) -> None:
        self.name = name

    def __call__(self, log: list[str]) -> int:
        log.append("case " + self.name)
        return 1


class Suite:
    def __init__(self, parts: list["Case | Suite"]) -> None:
        self.parts = parts

    def __call__(self, log: list[str]) -> int:
        total = 0
        for part in self.parts:
            total += part(log)
        return total


log: list[str] = []
top = Suite([Case("a"), Suite([Case("b"), Case("c")]), Case("d")])
print(top(log), log)
one: Case | Suite = Case("e")
print(one(log), log[-1])
