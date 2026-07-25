# probe: 上のインライン構築版の対照 -- 同じ形でレシーバだけ呼び出し由来にする。
#   これは通る。`c3de5e7` では 34 セル (入手経路 2 x 制御フロー 2 x 格納形 2 x
#   末尾 3、+ 内側クラスのフィールド数・保持側の追加フィールド・分岐前の読み出し・
#   200 反復・2 レシーバ同時・関数内・for・2 段ネストの 10 形) を測って
#   **exit 134 は 1 件も出ていない**。この形が abort する報告があれば、
#   ビルド (ブランチ + commit) の差か、記述と実際の綴りの差である。
# axes: acquire=call width=w1obj op=rebind flow=while read=none
# CLASSIFICATION: 1 正しい
# CPython 3.14 expects: 1


class Inner:
    def __init__(self, n: int) -> None:
        self.n: int = n


class Holder:
    def __init__(self, x: Inner) -> None:
        self.x: Inner = x


def mk() -> Holder:
    v: Inner = Inner(1)
    return Holder(v)


h = mk()
i = 0
while i < 2:
    y: Inner = Inner(2)
    h.x = y
    i = i + 1
print(1)
