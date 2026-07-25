# probe: borrowed list pop (shrink, no realloc)
# axes: acquire=param width=w3list op=pop flow=straight
# CLASSIFICATION: 3 loud 拒否 (診断)
#   /Users/user/Desktop/dev/lython/.claude/worktrees/agent-a121ace38a4d2e6ff/tests/probe/known_borrowed_list_pop.py:7:4: emit error: static type !py.contract<"builtins.list", [!py.contract<"builtins.int">]> does not provide 
# CPython 3.14 expects: 1

def drop(xs: list[int]) -> None:
    xs.pop()


xs: list[int] = [1, 2]
drop(xs)
print(len(xs))
