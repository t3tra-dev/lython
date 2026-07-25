# probe: flow_ifone_w3list.py と同一だが、末尾の読み出しを `print(len(o.f))` から
#   ローカルへの素の束縛に変えたもの。これが通るので、N7 の dominance 失敗には
#   「re-root されたレーンを合流後の呼び出しに押し込む読み出し」という第 2 の
#   必要条件がある (list フィールドに限る -- 下 2 件が対照)。
# axes: acquire=call width=w3list op=rebind flow=ifone read=plain-bind
# CLASSIFICATION: 1 正しい
# CPython 3.14 expects: (出力なし、正常終了)


class Box:
    def __init__(self, v: list[int]) -> None:
        self.f: list[int] = v


def mk() -> Box:
    v: list[int] = []
    return Box(v)


o = mk()
n = len("ab")
if n == 2:
    x: list[int] = [1, 2]
    o.f = x
read: list[int] = o.f
