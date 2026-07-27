# A loop-carried container rebuilt AND mutated on every iteration.
#
# The same shape as exception_loop_carried_rebuild, in the contracts that reach
# it through a mutation rather than through `__init__`: `LyList_ExtendM`,
# `LyList_SetSlice`, `LySet_AddBox`, `LySet_UpdateM`,
# `LySet_DifferenceUpdate` and `LyFrozenSet_Init` all declare
# `ly.ownership.transfer_args` on their receiver, so the mutation moves the
# token to a new name and the pre-transfer name rides the back edge. Every case
# below was refused with "released through a value already consumed by an
# ownership transfer" before the walk started dropping a stale name at the op
# that rebinds it.
#
# It is in a separate file from the exception spelling on purpose: these are the
# contracts rfc/lane-conversion-playbook.md schedules as "cascade confirmed", and
# a conversion that reintroduces the refusal should be able to see which family
# it broke without reading two failures as one.
def list_extend(n: int) -> int:
    cur: list[int] = [0]
    i = 0
    while i < n:
        cur = [i]
        cur.extend([i + 1, i + 2])
        i += 1
    return len(cur) + cur[0]


def list_slice(n: int) -> int:
    cur: list[int] = [0, 1, 2, 3]
    i = 0
    while i < n:
        cur = [i, i + 1, i + 2, i + 3]
        cur[1:3] = [9, 9, 9]
        i += 1
    return len(cur)


def list_append_past_capacity(n: int) -> int:
    cur: list[int] = [0]
    i = 0
    while i < n:
        cur = [i]
        j = 0
        while j < 200:
            cur.append(j)
            j += 1
        i += 1
    return len(cur)


def set_add(n: int) -> int:
    cur: set[int] = {0}
    i = 0
    while i < n:
        cur = {i}
        cur.add(i + 1)
        i += 1
    return len(cur)


def set_update(n: int) -> int:
    cur: set[int] = {0}
    i = 0
    while i < n:
        cur = {i}
        cur.update({i + 1, i + 2})
        i += 1
    return len(cur)


def set_difference_update(n: int) -> int:
    cur: set[int] = {0, 1, 2}
    i = 0
    while i < n:
        cur = {i, i + 1, i + 2}
        cur.difference_update({i})
        i += 1
    return len(cur)


def frozen(n: int) -> int:
    cur: frozenset[int] = frozenset([0])
    i = 0
    while i < n:
        cur = frozenset([i, i + 1])
        i += 1
    return len(cur)


print(list_extend(0))
print(list_extend(3))
print(list_slice(3))
print(list_append_past_capacity(3))
print(set_add(3))
print(set_update(3))
print(set_difference_update(3))
print(frozen(3))
