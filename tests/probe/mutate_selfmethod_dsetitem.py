# probe: in-place mutation through self inside a method -- dict field setitem (insert, may rehash)
# axes: acquire=self width=w3list/w1dict op=dsetitem flow=straight
# CLASSIFICATION @ kernel/4b fa71a3c: 1 正しい
# CPython 3.14 expects: 2 2

class Box:
    def __init__(self, v: dict[str, int]) -> None:
        self.f: dict[str, int] = v

    def touch(self) -> None:
        self.f["b"] = 2

    def show(self) -> None:
        print(len(self.f), self.f["b"])


def mk() -> Box:
    v: dict[str, int] = {"a": 1}
    return Box(v)


o = mk()
o.touch()
o.show()
