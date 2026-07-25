# probe: append to a list read out of a borrowed object's field
# axes: acquire=param+field width=w3list op=append flow=straight
# CLASSIFICATION @ kernel/4a 6c328b5: 1 正しい
# CPython 3.14 expects: 2

class Box:
    def __init__(self, v: list[int]) -> None:
        self.f: list[int] = v


def put(b: Box) -> None:
    b.f.append(2)


o = Box([1])
put(o)
print(len(o.f))
