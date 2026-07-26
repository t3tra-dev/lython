# probe: the alias-read shape with the mutation in a LOOP, crossing the
#   reallocation boundary several times. The loop matters on its own: the
#   relation "which slot names this entity" is per-SSA-value, and a merge that
#   carries evidence requires every arm to forward the same values -- which a
#   back edge cannot do once the payload has been reallocated. So the view
#   arrived at the mutation with the relation already gone, and the lowering
#   could neither publish the growth nor tell a loop-carried FIELD view from a
#   loop-carried local in order to refuse.
# axes: acquire=call width=w3list op=alias-grow-in-loop flow=loop
# CLASSIFICATION @ kernel/4b: 1 正しい
# CPython 3.14 expects: 101
# Observed at kernel/4a: SIGABRT in libsystem_malloc.
# Control: alias_grow_list_past_capacity_call.py (same growth, straight line) and
#   a plain loop-carried local list, which was never affected.


class Node:
    def __init__(self, v: list[int]) -> None:
        self.f: list[int] = v


def make() -> Node:
    v: list[int] = [1]
    return Node(v)


def touch(n: Node) -> None:
    ks: list[int] = n.f
    i: int = 0
    while i < 100:
        ks.append(i)
        i = i + 1


n = make()
touch(n)
print(len(n.f))
