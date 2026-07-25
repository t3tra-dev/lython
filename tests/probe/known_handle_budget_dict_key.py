# probe: REPORTED loud (budget 5): a 2-int NamedTuple-shaped class as a dict key
# axes: width=wNcls op=dict-key flow=straight
# CLASSIFICATION: 3 loud 拒否 (診断)
#   /Users/user/Desktop/dev/lython/.claude/worktrees/agent-a121ace38a4d2e6ff/tests/probe/known_handle_budget_dict_key.py:15:15: emit error: isinstance on an object-typed value requires dynamic object inspection, which is exc
# CPython 3.14 expects: 3

class Pair:
    def __init__(self, a: int, b: int) -> None:
        self.a: int = a
        self.b: int = b

    def __hash__(self) -> int:
        return self.a * 31 + self.b

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Pair):
            return NotImplemented
        return self.a == other.a and self.b == other.b


d: dict[Pair, int] = {}
d[Pair(1, 2)] = 3
print(d[Pair(1, 2)])
