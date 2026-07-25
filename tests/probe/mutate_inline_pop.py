# probe: in-place field mutation -- list field pop (shrink); receiver from inline
# axes: acquire=inline width=w3list/w1dict op=pop flow=straight
# CLASSIFICATION: 3 loud 拒否 (診断)
#   /Users/user/Desktop/dev/lython/.claude/worktrees/agent-a121ace38a4d2e6ff/tests/probe/mutate_inline_pop.py:18:0: emit error: static type !py.contract<"builtins.list", [!py.contract<"builtins.int">]> does not provide manif
# CPython 3.14 expects: 0

class Box:
    def __init__(self, v: list[int]) -> None:
        self.f: list[int] = v


def mk() -> Box:
    v: list[int] = [10]
    return Box(v)


v0: list[int] = [10]
o = Box(v0)
o.f.pop()
print(len(o.f))
