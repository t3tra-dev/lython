# probe: list フィールドを分岐内で再束縛して何も読まない。float / w1obj の
#   「読み出しなし」版と揃えるための 3 幅目 (通る)。
#   3 幅すべてで「読み出しなし = 通る」ため、**合流後の読み出しは全幅で必要条件**。
# axes: acquire=call width=w3list op=rebind flow=ifone read=none
# CLASSIFICATION @ kernel/4a 6c328b5: 1 正しい
# CPython 3.14 expects: (出力なし、正常終了)


class Box:
    def __init__(self, v: list[int]) -> None:
        self.f: list[int] = v


def mk() -> Box:
    v: list[int] = []
    return Box(v)


o = mk()
n = len("ab")
if n == 2:
    x: list[int] = [1, 2]
    o.f = x
