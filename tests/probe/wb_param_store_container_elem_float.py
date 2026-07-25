# probe: the callee stores into a field of an object it read out of a borrowed list
# axes: acquire=param+container width=float op=rebind flow=straight observe=writeback
# CLASSIFICATION @ kernel/integration 935280d: 1 正しい
# CPython 3.14 expects: 1.5

class Box:
    def __init__(self, v: float) -> None:
        self.f: float = v


def rebind(bs: list[Box]) -> None:
    fresh: float = 1.5
    bs[0].f = fresh


v: float = 0.0
boxes: list[Box] = [Box(v)]
rebind(boxes)
print(boxes[0].f)
