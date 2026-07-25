# probe: field rebind -- receiver read out of an except-bound exception's field; field type Wide
# axes: acquire=except width=wNcls op=rebind flow=straight
# CLASSIFICATION: 3 loud 拒否 (診断)
#   a '!py.contract<"Box">' value expands to 8 physical handles, but a payload box carries at most 5; it cannot be stored in a container slot or boxed field yet (reduce the class to fewer or narrower fields)
# CPython 3.14 expects: 1 2.5 z

class Other:
    def __init__(self, n: int) -> None:
        self.n: int = n


class Wide:
    def __init__(self, a: int, b: float, c: str) -> None:
        self.a: int = a
        self.b: float = b
        self.c: str = c


class Box:
    def __init__(self, v: Wide) -> None:
        self.f: Wide = v


class Err(Exception):
    def __init__(self, b: Box) -> None:
        super().__init__("boom")
        self.b: Box = b


def boom() -> None:
    v: Wide = Wide(0, 0.0, "")
    raise Err(Box(v))


try:
    boom()
except Err as e:
    o = e.b
    fresh: Wide = Wide(1, 2.5, "z")
    o.f = fresh
    print(o.f.a, o.f.b, o.f.c)
