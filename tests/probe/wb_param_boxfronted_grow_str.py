# probe: 同じく box-fronted な str フィールドを関数境界越しに「長さの違う値」で
#   置き換える (payload の再確保を伴う store が呼び出し元に見えるか)。
# axes: acquire=param width=w1str(box-fronted) op=rebind-longer flow=straight observe=writeback
# CLASSIFICATION @ kernel/4a 95cf6f7: 1 正しい
# CPython 3.14 expects: 32 0123456789abcdef0123456789abcdef


class Node:
    def __init__(self, s: str) -> None:
        self._s: str = s


def make() -> Node:
    v: str = "a"
    return Node(v)


def grow(n: Node) -> None:
    longer: str = "0123456789abcdef0123456789abcdef"
    n._s = longer


n = make()
grow(n)
print(len(n._s), n._s)
