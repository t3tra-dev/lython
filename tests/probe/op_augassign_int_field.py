# probe: augmented assignment to an int field of a call-obtained object
# axes: acquire=call width=w3int op=augassign flow=straight
# CLASSIFICATION @ kernel/integration 935280d: 1 正しい
# CPython 3.14 expects: 42

class Box:
    def __init__(self, v: int) -> None:
        self.f: int = v


def mk() -> Box:
    return Box(1)


o = mk()
o.f += 41
print(o.f)
