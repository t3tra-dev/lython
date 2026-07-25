# probe: REPORTED loud: a container-typed module global read from inside a function
# axes: op=module-global flow=straight
# CLASSIFICATION: 3 loud 拒否 (診断)
#   /Users/user/Desktop/dev/lython/.claude/worktrees/agent-a121ace38a4d2e6ff/tests/probe/known_container_module_global.py:10:11: emit error: unresolved name 'TABLE'
# CPython 3.14 expects: 1

TABLE: dict[str, int] = {"a": 1}


def look(k: str) -> int:
    return TABLE[k]


print(look("a"))
