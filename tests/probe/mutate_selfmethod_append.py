# probe: in-place mutation through self inside a method -- list field append (grow, may reallocate)
# axes: acquire=self width=w3list/w1dict op=append flow=straight
# CLASSIFICATION @ kernel/4b fa71a3c: 1 正しい
# CPython 3.14 expects: 2 10 20

class Box:
    def __init__(self, v: list[int]) -> None:
        self.f: list[int] = v

    def touch(self) -> None:
        self.f.append(20)

    def show(self) -> None:
        print(len(self.f), self.f[0], self.f[1])


def mk() -> Box:
    v: list[int] = [10]
    return Box(v)


o = mk()
o.touch()
o.show()
