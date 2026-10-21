from . import case


class Result:
    def __init__(self) -> None:
        self.seen: list[case.Case] = []

    def add(self, test: case.Case) -> None:
        self.seen.append(test)

    def names(self) -> list[str]:
        return [test.name for test in self.seen]
