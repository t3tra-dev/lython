# probe: box-fronted な dict フィールドを alias-grow-writeback で成長させる。
#   k-rfc 照会 B。657f0d8 ではローカル構築レシーバでも SIGABRT になった
#   (box のスロット identity は安定でも payload の realloc がレーンを stale に
#   するかどうかを見る形)。
# axes: acquire=inline width=w1dict(box-fronted) op=alias-grow-writeback flow=straight
# CLASSIFICATION @ kernel/4a 95cf6f7: 1 正しい
# CPython 3.14 expects: 2
# 観測: 8 回実行: SIGABRT 8 (libsystem_malloc)。libgmalloc: SIGSEGV


class Node:
    def __init__(self, m: dict[str, int]) -> None:
        self._m: dict[str, int] = m

    def add(self, k: str, v: int) -> None:
        d: dict[str, int] = self._m
        d[k] = v
        self._m = d


empty: dict[str, int] = {}
n = Node(empty)
n.add("a", 1)
n.add("b", 2)
print(len(n._m))
