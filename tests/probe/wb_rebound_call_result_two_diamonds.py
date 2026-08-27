# probe: an owned call result rebound by two sequential if-chains keeps its release
# axes: acquire=call width=int op=rebind flow=two-diamonds observe=regression
# CLASSIFICATION @ 2026-08-27: 1 正しい
#
# `length` holds an owned `int` from `len()` and each `if` chain rebinds the
# name. Every edge into the first merge BORROWS -- `length` is read again after
# it, so none of them can be a move -- and the merge was dropped for having no
# transfer, which left its argument neither owned nor lent. The pass then read
# the forward as though the token had moved, so `length` lost its release on the
# paths the merge does not carry it out on:
#
#   owned resource from @LyLong_FromI64 result 0 reaches function exit without
#   release, transfer, or owned return
#
# An all-borrow merge now takes an increment per edge and owns its argument, so
# both questions -- whether the source still has to be released, and whether an
# unwind out of the merge block needs a cleanup for it -- are answered the same
# way as everywhere else. One `if` chain compiled before; two did not, and
# neither did `_anchors` in runtime/lib/traceback.py, which is where this was
# found.
#
# CPython 3.14 expects: 13


def f(line: str, col: int, end_col: int) -> int:
    length = len(line)
    marker_end = length
    if end_col > col:
        marker_end = end_col
        if marker_end > length:
            marker_end = length
    if marker_end <= col:
        marker_end = col + 1
        if marker_end > length:
            marker_end = length
    return marker_end


print(f("return a // b", 7, 13))
