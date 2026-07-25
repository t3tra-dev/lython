# probe: 照会 B を call 由来レシーバで行った形。
# axes: acquire=call width=w1dict(box-fronted) op=alias-grow-writeback flow=straight
# CLASSIFICATION @ kernel/4a 6c328b5: 1 正しい
# CPython 3.14 expects: 2
# 観測: 8 回実行: SIGABRT 8 (libsystem_malloc)。libgmalloc: SIGSEGV


class Node:
    def __init__(self, m: dict[str, int]) -> None:
        self._m: dict[str, int] = m

    def add(self, k: str, v: int) -> None:
        d: dict[str, int] = self._m
        d[k] = v
        self._m = d


def leaf() -> Node:
    empty: dict[str, int] = {}
    return Node(empty)


n = leaf()
n.add("a", 1)
n.add("b", 2)
print(len(n._m))
