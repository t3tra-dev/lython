# probe: the value stored in a field is dropped from every other name, then read back
# axes: acquire=call width=w3list op=drop-other-names flow=straight
# CLASSIFICATION: 3 loud 拒否 (診断)
#   /Users/user/Desktop/dev/lython/.claude/worktrees/agent-a121ace38a4d2e6ff/tests/probe/op_del_field_value.py:19:4: emit error: `del fresh` is rejected (Lython deviation from CPython): locals are released when their scope e
# CPython 3.14 expects: 3 3

class Box:
    def __init__(self, v: list[int]) -> None:
        self.f: list[int] = v


def mk() -> Box:
    v: list[int] = [1]
    return Box(v)


o = mk()
fresh: list[int] = [1, 2, 3]
o.f = fresh
del fresh
print(len(o.f), o.f[2])
