# probe: in-place field mutation -- list field append (grow, may reallocate); receiver from container
# axes: acquire=container width=w3list/w1dict op=append flow=straight
# CLASSIFICATION: 3 loud 拒否 (診断)
#   loc(fused<{ly.source.end_col = 31 : i32, ly.source.end_line = 19 : i32, ly.source.start_col = 0 : i32, ly.source.start_line = 19 : i32}>["/Users/user/Desktop/dev/lython/.claude/worktrees/agent-a121ace38a4d2e6ff/tests/pro
# CPython 3.14 expects: 2 10 20

class Box:
    def __init__(self, v: list[int]) -> None:
        self.f: list[int] = v


def mk() -> Box:
    v: list[int] = [10]
    return Box(v)


boxes: list[Box] = [mk()]
o = boxes[0]
o.f.append(20)
print(len(o.f), o.f[0], o.f[1])
