# probe: callee stores into and reads back a borrowed receiver's int | None field, then the caller reads it
# axes: acquire=param width=optional op=rebind flow=straight observe=writeback-both
# CLASSIFICATION: 3 loud 拒否 (診断)
#   loc(fused<{ly.source.end_col = 28 : i32, ly.source.end_line = 20 : i32, ly.source.start_col = 4 : i32, ly.source.start_line = 20 : i32}>["/Users/user/Desktop/dev/lython/.claude/worktrees/agent-a121ace38a4d2e6ff/tests/pro
# CPython 3.14 expects: in callee: 5 / in caller: 5

class Box:
    def __init__(self, v: int | None) -> None:
        self.f: int | None = v


def mk() -> Box:
    v: int | None = None
    return Box(v)


def rebind(b: Box) -> None:
    fresh: int | None = 5
    b.f = fresh
    o = b
    print("in callee:", o.f)


o = mk()
rebind(o)
print("in caller:", o.f)
