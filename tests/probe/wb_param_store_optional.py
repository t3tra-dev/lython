# probe: callee stores into a borrowed receiver's int | None field; caller reads it back
# axes: acquire=param width=optional op=rebind flow=straight observe=writeback
# CLASSIFICATION: 3 loud 拒否 (診断)
#   loc(fused<{ly.source.end_col = 10 : i32, ly.source.end_line = 23 : i32, ly.source.start_col = 0 : i32, ly.source.start_line = 23 : i32}>["/Users/user/Desktop/dev/lython/.claude/worktrees/agent-a121ace38a4d2e6ff/tests/pro
# CPython 3.14 expects: 5

class Box:
    def __init__(self, v: int | None) -> None:
        self.f: int | None = v


def mk() -> Box:
    v: int | None = None
    return Box(v)


def rebind(b: Box) -> None:
    fresh: int | None = 5
    b.f = fresh


o = mk()
rebind(o)
print(o.f)
