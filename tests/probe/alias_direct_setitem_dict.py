# probe: 照会 B の対照 -- 同じ挿入をローカル別名を作らず `self._f["x"] = k` と
#   直接書く。これは通る (box のスロットを通した in-place 挿入)。
# axes: acquire=inline width=w1dict(box-fronted) op=direct-insert flow=straight
# CLASSIFICATION @ kernel/integration 935280d: 1 正しい
# CPython 3.14 expects: 1


class Node:
    def __init__(self, v: dict[str, int]) -> None:
        self._f: dict[str, int] = v

    def add(self, k: int) -> None:
        self._f["x"] = k


v0: dict[str, int] = {}
n = Node(v0)
n.add(1)
n.add(2)
print(len(n._f))
