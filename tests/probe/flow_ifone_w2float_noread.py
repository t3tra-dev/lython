# probe: 分岐内で float フィールドを再束縛し、**そのあと何も読まない**。
#   これが通ることが決定的である -- 再束縛だけでは dominance 失敗は起きない。
#   したがって「float は読み出し形に関係なく失敗する」という私の以前の要約は
#   誤りで、正しくは「合流後の読み出しは全幅で必要条件であり、幅が決めるのは
#   *どこまで安い読み出しなら再 root されたレーンを消費せずに済むか* の閾値」。
#   float には無料の読み出し形が存在しない (素の束縛でも materialise が要る)。
# axes: acquire=call width=w2float op=rebind flow=ifone read=none
# CLASSIFICATION @ kernel/integration 935280d: 1 正しい
# CPython 3.14 expects: (出力なし、正常終了)


class Box:
    def __init__(self, v: float) -> None:
        self.f: float = v


def mk() -> Box:
    v: float = 0.0
    return Box(v)


o = mk()
n = len("ab")
if n == 2:
    x: float = 1.5
    o.f = x
