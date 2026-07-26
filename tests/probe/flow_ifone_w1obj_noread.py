# probe: 上の float 版と同じ「読み出しなし」をユーザーオブジェクトフィールド
#   (ハンドル store 経路) で行う。これも通るので、再束縛だけでは失敗しないことが
#   3 幅すべてで確認できる。
# axes: acquire=call width=w1obj op=rebind flow=ifone read=none
# CLASSIFICATION @ kernel/4b fa71a3c: 1 正しい
# CPython 3.14 expects: (出力なし、正常終了)


class Other:
    def __init__(self, n: int) -> None:
        self.n: int = n


class Box:
    def __init__(self, v: Other) -> None:
        self.f: Other = v


def mk() -> Box:
    v: Other = Other(0)
    return Box(v)


o = mk()
n = len("ab")
if n == 2:
    x: Other = Other(7)
    o.f = x
