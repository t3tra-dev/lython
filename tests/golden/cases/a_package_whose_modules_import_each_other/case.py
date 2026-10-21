from . import result


class Case:
    def __init__(self, name: str) -> None:
        self.name = name

    def run(self, into: result.Result | None = None) -> result.Result:
        if into is None:
            into = result.Result()
        into.add(self)
        return into


class Loud(Case):
    def label(self) -> str:
        return self.name.upper()
