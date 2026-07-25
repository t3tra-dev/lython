# probe: a function rebinds a field of an object it created and returns that object
# axes: acquire=inline width=w3list op=return-from-function flow=straight
# CLASSIFICATION: 3 loud 拒否 (診断)
#   loc(fused<{ly.source.end_col = 12 : i32, ly.source.end_line = 20 : i32, ly.source.start_col = 4 : i32, ly.source.start_line = 20 : i32}>["/Users/user/Desktop/dev/lython/.claude/worktrees/agent-a121ace38a4d2e6ff/tests/pro
# CPython 3.14 expects: 3

class Box:
    def __init__(self, v: list[int]) -> None:
        self.f: list[int] = v


def mk() -> Box:
    v: list[int] = [1]
    return Box(v)


def build() -> Box:
    b = mk()
    fresh: list[int] = [1, 2, 3]
    b.f = fresh
    return b


o = build()
print(len(o.f))
