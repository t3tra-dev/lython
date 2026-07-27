# A container literal built inside a loop, holding the loop variable ITSELF.
#
# `for i in range(n): ys = [i]` was refused with
#
#   owned resource from @LyRangeIterator_Next result 0
#   reaches function exit without release, transfer, or owned return
#
# Execution is required, not just acceptance: the release this exercises is the
# one on the loop-EXIT edge, and the only difference between placing it there and
# placing it twice is a refcount that no static check on the accepted IR reports.
# `ys[0]` and `len(ys)` are read so the element's value is observed after the
# container took its reference, which is what a premature release corrupts.
#
# `@LyRangeIterator_Next` allocates its element UNCONDITIONALLY -- the manifest
# does `arith.select %valid, %current, %zero` and then calls `@LyLong_FromI64` on
# both paths -- so the element is owned even on the exhausted iteration. That is
# why `ly.runtime.valid_result_index` must NOT be read as a conditional-ownership
# marker: it says the element is MEANINGFUL, not that it is OWNED, and a
# conditional group would leak the exhausted path's element instead.
#
# The axes below are the ones that separated accept from refuse before the fix,
# established over 47 variants by two earlier readers plus 22 here. The element
# must be the iterator result ITSELF (`[i + 1]` always compiled, `k = i` then
# `[k]` did not); the container must be a list or a tuple (dict and set always
# compiled); and `[i, i + 1]` compiled while `[i, i]`, `[i, 7]` and `[7, i]` did
# not. `total += ys[0]` compiled while `total += len(ys)` did not. All of those
# are kept, because the fix is one placement decision and a regression in it
# would show up in an arbitrary-looking subset of them.
#
# `for ch in s` covers `@LyUnicodeStrIterator_Next`, a SECOND declaration with
# the same shape; that defect was predicted from a manifest census (exactly 3 of
# 1490 declarations carry `valid_result_index`) before it was run. The third,
# `@LyCounter_Next` (`lyrt.Counter`), is not spellable from Python source, so it
# rides the same code path without a case of its own.


def elem_list(n: int) -> int:
    for i in range(n):
        ys = [i]
    return 0


def elem_tuple(n: int) -> int:
    for i in range(n):
        ys = (i,)
    return 0


def elem_aliased(n: int) -> int:
    for i in range(n):
        k = i
        ys = [k]
    return 0


def elem_twice(n: int) -> int:
    for i in range(n):
        ys = [i, i]
    return 0


def elem_then_const(n: int) -> int:
    for i in range(n):
        ys = [i, 7]
    return 0


def const_then_elem(n: int) -> int:
    for i in range(n):
        ys = [7, i]
    return 0


def read_len(n: int) -> int:
    total = 0
    for i in range(n):
        ys = [i]
        total += len(ys)
    return total


def read_index(n: int) -> int:
    total = 0
    for i in range(n):
        ys = [i]
        total += ys[0]
    return total


def read_both(n: int) -> int:
    total = 0
    for i in range(n):
        ys = [i, i * 2]
        total += ys[0] + ys[1]
    return total


def elem_derived(n: int) -> int:
    total = 0
    for i in range(n):
        ys = [i + 1]
        total += ys[0]
    return total


def elem_dict(n: int) -> int:
    total = 0
    for i in range(n):
        ys = {i: 1}
        total += len(ys)
    return total


def elem_set(n: int) -> int:
    total = 0
    for i in range(n):
        ys = {i}
        total += len(ys)
    return total


def str_elem(s: str) -> int:
    total = 0
    for ch in s:
        ys = [ch]
        total += len(ys)
    return total


def str_elem_join(s: str) -> str:
    out = ""
    for ch in s:
        ys = [ch]
        out = out + ys[0]
    return out


def elem_break(n: int) -> int:
    total = 0
    for i in range(n):
        ys = [i]
        total += ys[0]
        if i > 1:
            break
    return total


def elem_empty_range() -> int:
    total = 0
    for i in range(0):
        ys = [i]
        total += ys[0]
    return total


print(elem_list(4))
print(elem_tuple(4))
print(elem_aliased(4))
print(elem_twice(4))
print(elem_then_const(4))
print(const_then_elem(4))
print(read_len(4))
print(read_index(4))
print(read_both(4))
print(elem_derived(4))
print(elem_dict(4))
print(elem_set(4))
print(str_elem("abcd"))
print(str_elem_join("abcd"))
print(elem_break(4))
print(elem_empty_range())
