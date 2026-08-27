# probe: a borrowed int parameter rebound into a local balances its lend
# axes: acquire=param width=int op=alias flow=branchy observe=regression
# CLASSIFICATION @ 2026-08-28: 1 正しい
#
# `start = col` and `marker_end = end_col` copy a BORROWED parameter into a
# local, and each takes a borrow-to-own lend. Two things lost the returns:
#
#   1. the walk kept a PRE-RENAME name verbatim across edges, so the release
#      written under the loop's name for a naming taken before the loop was
#      invisible and the balance climbed one lend per rename;
#   2. a `cond_br` that forwards one value to BOTH successors' arguments made
#      the candidate propagation give up ("group split across successors"), so
#      the loop header's own arguments were never an owned group -- and the
#      loop-exit edge then LENT them instead of transferring, a lend nothing
#      returns.
#
# Either rebind alone is fine, and so is this body without the loop between
# them: what it takes is a lend across a rename and a merge fed twice by one
# branch. Refused as "borrowed entry argument 1 of @_anchors reaches function
# exit with 1 retained ownership token(s)", then argument 2 for the other
# rebind, over IR whose lends and returns pair exactly except for (2).
#
# This is `_anchors` in runtime/lib/traceback.py, which is written this way
# again now that it compiles.
#
# CPython 3.14 expects:
#        ~~^^~~


def _is_operator_char(ch: str) -> bool:
    return (ch == "+" or ch == "-" or ch == "*" or ch == "/" or ch == "%"
            or ch == "@" or ch == "&" or ch == "|" or ch == "^" or ch == "<"
            or ch == ">")


def _anchors(line: str, col: int, end_col: int, mode: int) -> str:
    length = len(line)
    if length == 0:
        return ""
    start = 0
    if col > 0 and col < length:
        start = col
    else:
        while start < length and (line[start] == " " or line[start] == "\t"):
            start += 1
    if start >= length:
        return ""
    marker_end = length
    if end_col > col and end_col > 0:
        marker_end = end_col
        if marker_end > length:
            marker_end = length
    if marker_end <= start:
        marker_end = start + 1
        if marker_end > length:
            marker_end = length

    caret_start = -1
    caret_end = -1
    if mode != 2:
        split = start
        while split < marker_end:
            if line[split] == "(" or line[split] == "[":
                caret_start = split
                caret_end = marker_end
                break
            split += 1
        if caret_start < 0:
            op = start
            while op < marker_end:
                if _is_operator_char(line[op]):
                    run = op + 1
                    while run < marker_end and _is_operator_char(line[run]):
                        run += 1
                    caret_start = op
                    caret_end = run
                    break
                op += 1
    if caret_start < 0:
        if start == 0 and marker_end >= length:
            return ""
        caret_start = start
        caret_end = marker_end

    out = ""
    pad = 0
    while pad < start:
        if line[pad] == "\t":
            out += "\t"
        else:
            out += " "
        pad += 1
    mark = start
    while mark < marker_end:
        if caret_start <= mark and mark < caret_end:
            out += "^"
        else:
            out += "~"
        mark += 1
    return out


print(_anchors("return a // b", 7, 13, 1))
