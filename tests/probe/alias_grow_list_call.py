# probe: フィールドの list をローカルに束縛して in-place で成長させ、そのローカルを
#   フィールドに書き戻す (call 由来レシーバ)。k-rfc 照会 A。
# axes: acquire=call width=w3list op=alias-grow-writeback flow=straight
# CLASSIFICATION @ kernel/4a 6c328b5: 1 正しい
# CPython 3.14 expects: 2
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
print(len(n._kids))
