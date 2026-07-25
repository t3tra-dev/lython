# probe: in-place field mutation inside a while loop (append)
# axes: acquire=call width=w3list/w1dict op=append flow=while
# CLASSIFICATION: 3 loud 拒否 (診断)
#   loc(fused<{ly.source.end_col = 17 : i32, ly.source.end_line = 19 : i32, ly.source.start_col = 4 : i32, ly.source.start_line = 19 : i32}>["/Users/user/Desktop/dev/lython/.claude/worktrees/agent-a121ace38a4d2e6ff/tests/pro
# CPython 3.14 expects: 3

class Box:
    def __init__(self, v: list[int]) -> None:
        self.f: list[int] = v


def mk() -> Box:
    v: list[int] = []
    return Box(v)


o = mk()
k = 0
while k < 3:
    o.f.append(k)
    k = k + 1
print(len(o.f))
