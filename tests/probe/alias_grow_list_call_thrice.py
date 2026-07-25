# probe: 照会 A を 3 回に増やした形 (657f0d8 では print の前に abort した)。
#   繰り返し回数が abort の位置を動かすかを見る。
# axes: acquire=call width=w3list op=alias-grow-writeback flow=straight repeats=3
# CLASSIFICATION @ kernel/4a 6c328b5: 1 正しい
# CPython 3.14 expects: 3
# 観測: 8 回実行: SIGTRAP 3 / abort(refcount) 5。libgmalloc: SIGSEGV (決定的)


class Node:
    def __init__(self, kids: list[int]) -> None:
        self._kids: list[int] = kids

    def add(self, v: int) -> None:
        ks: list[int] = self._kids
        ks.append(v)
        self._kids = ks


def leaf() -> Node:
    empty: list[int] = []
    return Node(empty)


n = leaf()
n.add(1)
n.add(2)
n.add(3)
print(len(n._kids))
