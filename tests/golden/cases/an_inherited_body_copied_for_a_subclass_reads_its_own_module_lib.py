_OPEN: str = "<"


def _decorate(text: str) -> str:
    return _OPEN + text + ">"


class Base:
    def name(self) -> str:
        return "base"

    def run(self) -> str:
        return _decorate(self.name())

    def __repr__(self) -> str:
        return "Base" + _decorate(self.name())
