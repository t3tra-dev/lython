# probe: 照会 A の分解の対照 -- ローカルに読み出して**変異させずに**書き戻す。
#   これが通るので、読み出し + 書き戻しの組そのものは無害だと決まる。
# axes: acquire=call width=w3list op=alias-read+writeback(no mutation) flow=straight
# CLASSIFICATION @ kernel/4a 6c328b5: 1 正しい
# CPython 3.14 expects: 0


class Node:
    def __init__(self, v: list[int]) -> None:
        self._f: list[int] = v

    def add(self, k: int) -> None:
        ks: list[int] = self._f
        self._f = ks


def leaf() -> Node:
    v: list[int] = []
    return Node(v)


n = leaf()
n.add(1)
n.add(2)
print(len(n._f))
