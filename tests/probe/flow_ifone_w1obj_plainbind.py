# probe: 同じ対照のユーザーオブジェクト版 (ハンドル store 経路)。素の束縛でも
#   dominance 失敗のまま。float 版と合わせて「素の束縛で通るのは list だけ」を
#   確定させる 2 件目。
# axes: acquire=call width=w1obj op=rebind flow=ifone read=plain-bind
# CLASSIFICATION: 3 loud 拒否 (MLIR verifier 失敗 = 最早境界での診断になっていない)
#   loc(fused<{ly.source.end_col = 4 : i32, ly.source.end_line = 29 : i32, ly.source.start_col = 0 : i32, ly.source.start_line = 29 : i32}>["/Users/user/Desktop/dev/lython/.claude/worktrees/agent-a121ace38a4d2e6ff/tests/prob
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
