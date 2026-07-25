# probe: フィールドの list をローカルに読み出し、そのローカルを in-place で
#   成長させるだけ (**書き戻しをしない**)。照会 A の分解 -- これが落ちるので
#   `self._f = ks` の書き戻しは真因ではないことが決まる。
# axes: acquire=call width=w3list op=alias-read+grow(no writeback) flow=straight
# CLASSIFICATION @ kernel/4a 95cf6f7: 1 正しい
# CPython 3.14 expects: 2
# 観測: 8 回実行: silent 3 (`1` を出力) / abort(refcount) 4 / signal 1。libgmalloc: SIGSEGV (決定的)


class Node:
    def __init__(self, v: list[int]) -> None:
        self._f: list[int] = v

    def add(self, k: int) -> None:
        ks: list[int] = self._f
        ks.append(k)


def leaf() -> Node:
    v: list[int] = []
    return Node(v)


n = leaf()
n.add(1)
n.add(2)
print(len(n._f))
