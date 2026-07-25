# probe: a method stores into self's int | None field; the caller reads it back
# axes: acquire=self width=optional op=rebind flow=straight observe=writeback
# CLASSIFICATION: 3 loud 拒否 (診断)
#   loc(fused<{ly.source.end_col = 10 : i32, ly.source.end_line = 22 : i32, ly.source.start_col = 0 : i32, ly.source.start_line = 22 : i32}>["/Users/user/Desktop/dev/lython/.claude/worktrees/agent-a121ace38a4d2e6ff/tests/pro
# CPython 3.14 expects: 5

class Box:
    def __init__(self, v: int | None) -> None:
        self.f: int | None = v

    def set(self) -> None:
        fresh: int | None = 5
        self.f = fresh


def mk() -> Box:
    v: int | None = None
    return Box(v)


o = mk()
o.set()
print(o.f)
