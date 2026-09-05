# What: a nested `def` inside a METHOD that declares one of the method's locals
# `nonlocal`. An inlined method body is still a function body, and the local has
# to become a shared cell for the nested function to write through it. Runtime
# values, because the question is whether the write is seen by the method after
# the nested call returns.


class Counter:
    def __init__(self, step: int) -> None:
        self.step = step

    def run(self, n: int) -> int:
        total = 0

        def bump(v: int) -> None:
            nonlocal total
            total = total + v * self.step

        for i in range(n):
            bump(i)
        return total

    def label(self) -> str:
        text = ""

        def add(part: str) -> None:
            nonlocal text
            text = text + part

        add("a")
        add("b")
        return text

    @staticmethod
    def counted(n: int) -> int:
        seen = 0

        def tick() -> None:
            nonlocal seen
            seen += 1

        for _ in range(n):
            tick()
        return seen

    def captured(self) -> int:
        # Capture by READ, which worked before and must keep working.
        base = self.step * 10

        def read() -> int:
            return base

        return read()


print(Counter(2).run(4), Counter(3).run(4))
print(Counter(1).label())
print(Counter.counted(5))
print(Counter(7).captured())
