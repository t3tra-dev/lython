# probe: field rebind reached through with control flow; field type str
# axes: acquire=call width=w1str op=rebind flow=with
# CLASSIFICATION: 3 loud 拒否 (診断)
#   loc(fused<{ly.source.end_col = 10 : i32, ly.source.end_line = 25 : i32, ly.source.start_col = 5 : i32, ly.source.start_line = 25 : i32}>["/Users/user/Desktop/dev/lython/.claude/worktrees/agent-a121ace38a4d2e6ff/tests/pro
# CPython 3.14 expects: 2

class Box:
    def __init__(self, v: str) -> None:
        self.f: str = v


def mk() -> Box:
    v: str = ""
    return Box(v)


class Ctx:
    def __enter__(self) -> "Ctx":
        return self

    def __exit__(self, a: object, b: object, c: object) -> bool:
        return False


o = mk()
with Ctx():
    x: str = "ab"
    o.f = x
print(len(o.f))
