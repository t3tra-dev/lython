# probe: ユーザーオブジェクトフィールドを while ループで再束縛し、読み出しなし。
#   ただしレシーバは**インライン構築** (`Holder(Inner(1))`)。dominance 失敗になる。
#   `flow_while_w1obj_callrecv_noread.py` (レシーバが呼び出し由来 = 通る) が対照で、
#   この 2 件が追補 8 の重要な限定を示す:
#   **「読み出しなしなら通る」はレシーバが呼び出し由来のときだけ成立する。**
#   オブジェクトフィールドでインライン構築レシーバの場合、末尾に何を置いても
#   (何も置かない / 定数の print / フィールド読み) 12 セルすべて dominance 失敗。
#   つまりこの幅では入手経路の軸が読み出し形の軸より優位である。
# axes: acquire=inline width=w1obj op=rebind flow=while read=none
# CLASSIFICATION @ kernel/4a 95cf6f7: 1 正しい
# CPython 3.14 expects: 1


class Inner:
    def __init__(self, n: int) -> None:
        self.n: int = n


class Holder:
    def __init__(self, x: Inner) -> None:
        self.x: Inner = x


h = Holder(Inner(1))
i = 0
while i < 2:
    y: Inner = Inner(2)
    h.x = y
    i = i + 1
print(1)
