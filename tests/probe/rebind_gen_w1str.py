# probe: field rebind -- receiver from a generator yield; field type str
# axes: acquire=gen width=w1str op=rebind flow=straight
# CLASSIFICATION: 3 loud 拒否 (診断)
#   loc(fused<{ly.source.end_col = 14 : i32, ly.source.end_line = 23 : i32, ly.source.start_col = 9 : i32, ly.source.start_line = 23 : i32}>["/Users/user/Desktop/dev/lython/.claude/worktrees/agent-a121ace38a4d2e6ff/tests/pro
# CPython 3.14 expects: 4 abcd

class Box:
    def __init__(self, v: str) -> None:
        self.f: str = v


def mk() -> Box:
    v: str = ""
    return Box(v)


from typing import Iterator


def gen() -> Iterator[Box]:
    yield mk()


for o in gen():
    fresh: str = "abcd"
    o.f = fresh
    print(len(o.f), o.f)
