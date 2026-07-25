# probe: list 版の対照 -- 同じ「素の束縛」に変えても float フィールドは dominance
#   失敗のままである。ただし理由は「読み出し形と無関係」ではない:
#   `flow_ifone_w2float_noread.py` (読み出しを消すと通る) と合わせると、
#   **合流後の読み出しは全幅で必要条件**であり、幅が決めるのは
#   *どこまで安い読み出しなら再 root されたレーンを消費せずに済むか* の閾値。
#   list は素の束縛が純粋な SSA コピーなので無料、float は素の束縛でも f64 を
#   materialise するので無料の読み出し形が存在しない。
# axes: acquire=call width=w2float op=rebind flow=ifone read=plain-bind
# CLASSIFICATION: 3 loud 拒否 (MLIR verifier 失敗 = 最早境界での診断になっていない)
#   operand #0 does not dominate this use
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
read: float = o.f
