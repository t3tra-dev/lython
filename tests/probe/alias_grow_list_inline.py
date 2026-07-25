# probe: 照会 A の対照 -- 同じ alias-grow-writeback をローカル構築レシーバで行う
#   (657f0d8 では call 由来だけ abort し、こちらは正常終了した = marker gap の証拠)。
# axes: acquire=inline width=w3list op=alias-grow-writeback flow=straight
# CLASSIFICATION: 1 正しい
# CPython 3.14 expects: 2
# 観測: 8 回実行: 正常終了 8。libgmalloc: 正常終了


class Node:
    def __init__(self, kids: list[int]) -> None:
        self._kids: list[int] = kids

    def add(self, v: int) -> None:
        ks: list[int] = self._kids
        ks.append(v)
        self._kids = ks


empty: list[int] = []
n = Node(empty)
n.add(1)
n.add(2)
print(len(n._kids))
