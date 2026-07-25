# probe: box-fronted フィールドの payload を関数境界越しに realloc させる
#   (借用レシーバの dict フィールドに callee が挿入する)。box のスロット identity
#   が安定なら見えるはずだが、payload の realloc がレーンを stale にするなら
#   崩れる。照会 B の「安定スロットだけで足りるか」の判定材料。
# axes: acquire=param width=w1dict(box-fronted) op=inplace-insert flow=straight observe=writeback
# CLASSIFICATION @ kernel/integration 935280d: 1 正しい
# CPython 3.14 expects: 3 3


class Node:
    def __init__(self, m: dict[str, int]) -> None:
        self._m: dict[str, int] = m


def make() -> Node:
    empty: dict[str, int] = {}
    return Node(empty)


def put(n: Node, k: str, v: int) -> None:
    n._m[k] = v


n = make()
put(n, "a", 1)
put(n, "b", 2)
put(n, "c", 3)
print(len(n._m), n._m["c"])
