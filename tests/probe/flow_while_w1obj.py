# probe: field rebind reached through while control flow; field type Other
# axes: acquire=call width=w1obj op=rebind flow=while
# CLASSIFICATION: 3 loud 拒否 (MLIR verifier 失敗 = 最早境界での診断になっていない)
#   loc(fused<{ly.source.end_col = 11 : i32, ly.source.end_line = 27 : i32, ly.source.start_col = 6 : i32, ly.source.start_line = 27 : i32}>["/Users/user/Desktop/dev/lython/.claude/worktrees/agent-a121ace38a4d2e6ff/tests/pro
# CPython 3.14 expects: 7

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
k = 0
while k < 2:
    x: Other = Other(7)
    o.f = x
    k = k + 1
print(o.f.n)
