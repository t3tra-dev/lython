# probe: 照会 B の分解 -- box-fronted な dict フィールドをローカルに読み出し、
#   そのローカルに挿入するだけ (**書き戻しをしない**)。これも落ちるので、
#   B の真因も書き戻しではなく「別名を再確保させる変異」だと決まる。
#   直接 `self._f["x"] = k` と書けば通る (alias_direct_setitem_dict.py) 点が対照。
# axes: acquire=inline width=w1dict(box-fronted) op=alias-read+insert(no writeback) flow=straight
# CLASSIFICATION: 4 クラッシュ / abort
#   libsystem_malloc フレームを含む; exit -6
# CPython 3.14 expects: 1
# 観測: 8 回実行: SIGABRT 8 (libsystem_malloc)。libgmalloc: SIGSEGV


class Node:
    def __init__(self, v: dict[str, int]) -> None:
        self._f: dict[str, int] = v

    def add(self, k: int) -> None:
        d: dict[str, int] = self._f
        d["x"] = k


v0: dict[str, int] = {}
n = Node(v0)
n.add(1)
n.add(2)
print(len(n._f))
