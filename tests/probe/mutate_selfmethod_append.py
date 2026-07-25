# probe: in-place mutation through self inside a method -- list field append (grow, may reallocate)
# axes: acquire=self width=w3list/w1dict op=append flow=straight
# CLASSIFICATION: 3 loud 拒否 (診断)
#   owned resource from @LyLong_Repr result 0 is released or transferred more than once on one CFG path
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
