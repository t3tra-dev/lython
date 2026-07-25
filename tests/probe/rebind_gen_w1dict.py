# probe: field rebind -- receiver from a generator yield; field type dict[str, int]
# axes: acquire=gen width=w1dict op=rebind flow=straight
# CLASSIFICATION: 3 loud 拒否 (診断)
#   loc(fused<{ly.source.end_col = 14 : i32, ly.source.end_line = 23 : i32, ly.source.start_col = 9 : i32, ly.source.start_line = 23 : i32}>["/Users/user/Desktop/dev/lython/.claude/worktrees/agent-a121ace38a4d2e6ff/tests/pro
# CPython 3.14 expects: 2 1

class Box:
    def __init__(self, v: dict[str, int]) -> None:
        self.f: dict[str, int] = v


def mk() -> Box:
    v: dict[str, int] = {}
    return Box(v)


from typing import Iterator


def gen() -> Iterator[Box]:
    yield mk()


for o in gen():
    fresh: dict[str, int] = {"a": 1, "b": 2}
    o.f = fresh
    print(len(o.f), o.f["a"])
