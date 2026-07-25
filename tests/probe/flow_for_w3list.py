# probe: field rebind reached through for control flow; field type list[int]
# axes: acquire=call width=w3list op=rebind flow=for
# CLASSIFICATION: 3 loud 拒否 (MLIR verifier 失敗 = 最早境界での診断になっていない)
#   operand #0 does not dominate this use
# CPython 3.14 expects: 2

class Box:
    def __init__(self, v: list[int]) -> None:
        self.f: list[int] = v


def mk() -> Box:
    v: list[int] = []
    return Box(v)


o = mk()
for _ in range(2):
    x: list[int] = [1, 2]
    o.f = x
print(len(o.f))
