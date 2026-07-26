# probe: the alias-read shape ACROSS THE REALLOCATION BOUNDARY. Identical to
#   alias_grow_list_call.py except that the list starts full: a container is
#   allocated with capacity 64, so a shorter list's append reallocates nothing
#   and the shape only tests an in-place write. 65 elements make the single
#   append grow the payload, which is the case the 12-cell grid could not see --
#   capacity is not one of its axes.
# axes: acquire=call width=w3list op=alias-grow-past-capacity flow=straight
# CLASSIFICATION @ kernel/4b: 1 正しい
# CPython 3.14 expects: 66 / 66
# Observed at kernel/4a: SIGABRT in libsystem_malloc (LyList_EnsureCapacity
#   transfers the container token, and a borrowed interior view has none to give,
#   so the transfer consumed the field slot's reference).
# Control: alias_grow_list_call.py, the same program under capacity.


class Node:
    def __init__(self, v: list[int]) -> None:
        self.f: list[int] = v


def make() -> Node:
    v: list[int] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64]
    return Node(v)


n = make()
ks: list[int] = n.f
ks.append(999)
print(len(n.f))
print(len(ks))
