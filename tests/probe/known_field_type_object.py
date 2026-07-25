# probe: REPORTED loud: a class field holding a type object
# axes: width=type op=field flow=straight
# CLASSIFICATION: 3 loud 拒否 (診断)
#   /Users/user/Desktop/dev/lython/.claude/worktrees/agent-a121ace38a4d2e6ff/tests/probe/known_field_type_object.py:17:6: emit error: static type !py.contract<"Box"> does not provide manifest method 't'
# CPython 3.14 expects: 5

class Other:
    def __init__(self, n: int) -> None:
        self.n: int = n


class Box:
    def __init__(self, t: type[Other]) -> None:
        self.t: type[Other] = t


o = Box(Other)
print(o.t(5).n)
