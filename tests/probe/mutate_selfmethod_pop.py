# probe: in-place mutation through self inside a method -- list field pop (shrink)
# axes: acquire=self width=w3list/w1dict op=pop flow=straight
# CLASSIFICATION @ kernel/4a 6c328b5: 3 loud 拒否 (診断)
#   static type !py.contract<"builtins.list", [!py.contract<"builtins.int">]> does not provide manifest method 'pop'
# CPython 3.14 expects: 0

class Box:
    def __init__(self, v: list[int]) -> None:
        self.f: list[int] = v

    def touch(self) -> None:
        self.f.pop()

    def show(self) -> None:
        print(len(self.f))


def mk() -> Box:
    v: list[int] = [10]
    return Box(v)


o = mk()
o.touch()
o.show()
