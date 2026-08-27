# probe: an owned call result rebound by two sequential if-chains loses its release
# axes: acquire=call width=int op=rebind flow=two-diamonds observe=refusal
# CLASSIFICATION @ 2026-08-27: 3 誤って拒否する
#
# `length` holds an owned `int` from `len()`. Each `if` chain rebinds the name
# and merges; the second chain's merge is reached by TWO edges from ONE
# terminator carrying DIFFERENT operands, which is the shape
# insertBlockArgMergeBorrowRetains declines outright ("Two edges into the merge
# from one terminator cannot both be retained (only one is taken at runtime)").
# Declining removes the merge candidate, and with it the release of the value
# the return did not take:
#
#   owned resource from @LyLong_FromI64 result 0 reaches function exit without
#   release, transfer, or owned return
#
# One chain compiles. Two chains compile when `length` is a PARAMETER rather
# than a call result. Both chains together on a call result do not.
#
# Found while making traceback.py's anchor columns match CPython; the affected
# code is now runtime/lib/traceback.py's `_marker_end`, written as its own
# function so the answer is bound once.
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
