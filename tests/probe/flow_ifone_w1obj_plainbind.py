# probe: 同じ対照のユーザーオブジェクト版 (ハンドル store 経路)。素の束縛でも
#   dominance 失敗のまま = 「素の束縛で通るのは list だけ」を確定させる 2 件目。
#   ただし `flow_ifone_w1obj_noread.py` (読み出しを消すと通る) があるので、
#   再束縛だけでは失敗しない -- 幅が決めるのは無料の読み出し形が存在するか否か。
# axes: acquire=call width=w1obj op=rebind flow=ifone read=plain-bind
# CLASSIFICATION @ kernel/integration 935280d: 1 正しい
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
read: Other = o.f
