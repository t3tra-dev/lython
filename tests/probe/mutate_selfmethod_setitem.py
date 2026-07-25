# probe: in-place mutation through self inside a method -- list field setitem (in place, no realloc)
# axes: acquire=self width=w3list/w1dict op=setitem flow=straight
# CLASSIFICATION @ kernel/4a 95cf6f7: 1 正しい
# CPython 3.14 expects: 1 99

class Box:
    def __init__(self, v: list[int]) -> None:
        self.f: list[int] = v

    def touch(self) -> None:
        self.f[0] = 99

    def show(self) -> None:
        print(len(self.f), self.f[0])


def mk() -> Box:
    v: list[int] = [10]
    return Box(v)


o = mk()
o.touch()
o.show()
