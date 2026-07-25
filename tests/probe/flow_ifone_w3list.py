# probe: field rebind reached through ifone control flow; field type list[int]
# axes: acquire=call width=w3list op=rebind flow=ifone
# CLASSIFICATION: 3 loud 拒否 (MLIR verifier 失敗 = 最早境界での診断になっていない)
#   loc(fused<{ly.source.end_col = 14 : i32, ly.source.end_line = 21 : i32, ly.source.start_col = 6 : i32, ly.source.start_line = 21 : i32}>["/Users/user/Desktop/dev/lython/.claude/worktrees/agent-a121ace38a4d2e6ff/tests/pro
# CPython 3.14 expects: 2

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
print(len(o.f))
