# probe: in-place mutation through self inside a method -- dict field del (in place)
# axes: acquire=self width=w3list/w1dict op=ddel flow=straight
# CLASSIFICATION @ kernel/4b fa71a3c: 1 正しい
# CPython 3.14 expects: 0

class Box:
    def __init__(self, v: dict[str, int]) -> None:
        self.f: dict[str, int] = v

    def touch(self) -> None:
        del self.f["a"]

    def show(self) -> None:
        print(len(self.f))


def mk() -> Box:
    v: dict[str, int] = {"a": 1}
    return Box(v)


o = mk()
o.touch()
o.show()
