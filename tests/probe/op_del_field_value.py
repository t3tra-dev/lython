# probe: the value stored in a field is dropped from every other name, then read back
# axes: acquire=call width=w3list op=drop-other-names flow=straight
# CLASSIFICATION @ kernel/4a 95cf6f7: 3 loud 拒否 (診断)
#   `del fresh` is rejected (Lython deviation from CPython): locals are released when their scope ends, so deleting a variable is unnecessary
# CPython 3.14 expects: 3 3

class Box:
    def __init__(self, v: list[int]) -> None:
        self.f: list[int] = v


def mk() -> Box:
    v: list[int] = [1]
    return Box(v)


o = mk()
fresh: list[int] = [1, 2, 3]
o.f = fresh
del fresh
print(len(o.f), o.f[2])
