# probe: REPORTED loud: a container-typed module global read from inside a function
# axes: op=module-global flow=straight
# CLASSIFICATION @ kernel/4a 95cf6f7: 3 loud 拒否 (診断)
#   unresolved name 'TABLE'
# CPython 3.14 expects: 1

TABLE: dict[str, int] = {"a": 1}


def look(k: str) -> int:
    return TABLE[k]


print(look("a"))
