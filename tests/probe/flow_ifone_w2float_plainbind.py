# probe: 上の list 版の対照 -- 同じ「素の束縛」に変えても float フィールドは
#   dominance 失敗のままである。読み出し形が第 2 の必要条件になるのは 3 レーンの
#   list に限り、**レーン re-root の代表幅である float では読み出し形に関係なく
#   失敗する**。したがって N7 は「読み出し形の問題」に置き換えられるのではなく、
#   幅の軸に読み出し形の軸が list の行だけ重なる、という構造になる。
# axes: acquire=call width=w2float op=rebind flow=ifone read=plain-bind
# CLASSIFICATION: 3 loud 拒否 (MLIR verifier 失敗 = 最早境界での診断になっていない)
#   loc(fused<{ly.source.end_col = 4 : i32, ly.source.end_line = 26 : i32, ly.source.start_col = 0 : i32, ly.source.start_line = 26 : i32}>["/Users/user/Desktop/dev/lython/.claude/worktrees/agent-a121ace38a4d2e6ff/tests/prob
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
