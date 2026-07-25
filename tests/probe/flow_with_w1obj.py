# probe: field rebind reached through with control flow; field type Other
# axes: acquire=call width=w1obj op=rebind flow=with
# CLASSIFICATION: 3 loud 拒否 (診断)
#   loc(fused<{ly.source.end_col = 10 : i32, ly.source.end_line = 30 : i32, ly.source.start_col = 5 : i32, ly.source.start_line = 30 : i32}>["/Users/user/Desktop/dev/lython/.claude/worktrees/agent-a121ace38a4d2e6ff/tests/pro
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


class Ctx:
    def __enter__(self) -> "Ctx":
        return self

    def __exit__(self, a: object, b: object, c: object) -> bool:
        return False


o = mk()
with Ctx():
    x: Other = Other(7)
    o.f = x
print(o.f.n)
