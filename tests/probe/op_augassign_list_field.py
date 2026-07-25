# probe: augmented assignment (+=) to a list field of a call-obtained object
# axes: acquire=call width=w3list op=augassign flow=straight
# CLASSIFICATION: 3 loud 拒否 (診断)
#   loc(fused<{ly.source.end_col = 3 : i32, ly.source.end_line = 17 : i32, ly.source.start_col = 0 : i32, ly.source.start_line = 17 : i32}>["/Users/user/Desktop/dev/lython/.claude/worktrees/agent-a121ace38a4d2e6ff/tests/prob
# CPython 3.14 expects: 3 3

class Box:
    def __init__(self, v: list[int]) -> None:
        self.f: list[int] = v


def mk() -> Box:
    v: list[int] = [1]
    return Box(v)


o = mk()
o.f += [2, 3]
print(len(o.f), o.f[2])
