# probe: 照会 A の分解の対照 -- ローカル別名への変異を、成長する `append` ではなく
#   再確保を伴わない `setitem` にする。これが通るので、引き金は「別名の変異」
#   一般ではなく**別名を再確保させる変異**だと決まる。
# axes: acquire=call width=w3list op=alias-read+setitem(no realloc) flow=straight
# CLASSIFICATION @ kernel/4a 95cf6f7: 1 正しい
# CPython 3.14 expects: 1


class Node:
    def __init__(self, v: list[int]) -> None:
        self._f: list[int] = v

    def add(self, k: int) -> None:
        ks: list[int] = self._f
        ks[k] = k
        self._f = ks


def leaf() -> Node:
    v: list[int] = [0, 0, 0]
    return Node(v)


n = leaf()
n.add(1)
n.add(2)
print(n._f[1])
