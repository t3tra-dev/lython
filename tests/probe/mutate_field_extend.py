# probe: in-place field mutation -- list field extend; receiver from field
# axes: acquire=field width=w3list/w1dict op=extend flow=straight
# CLASSIFICATION: 3 loud 拒否 (診断)
#   loc(fused<{ly.source.end_col = 18 : i32, ly.source.end_line = 23 : i32, ly.source.start_col = 0 : i32, ly.source.start_line = 23 : i32}>["/Users/user/Desktop/dev/lython/.claude/worktrees/agent-a121ace38a4d2e6ff/tests/pro
# CPython 3.14 expects: 3 3

class Box:
    def __init__(self, v: list[int]) -> None:
        self.f: list[int] = v


def mk() -> Box:
    v: list[int] = [10]
    return Box(v)


class Holder:
    def __init__(self, b: Box) -> None:
        self.inner: Box = b


h = Holder(mk())
o = h.inner
o.f.extend([2, 3])
print(len(o.f), o.f[2])
