# probe: a borrowed int parameter aliased into a local keeps a token at exit
# axes: acquire=param width=int op=alias flow=branchy observe=refusal
# CLASSIFICATION @ 2026-08-27: 3 誤って拒否する
#
# `start = col` and `marker_end = end_col` copy a BORROWED parameter into a
# local. Each takes a borrow-to-own retain, and in a function this branchy the
# walk cannot find where the frame gives them back:
#
#   borrowed entry argument 1 of @_anchors reaches function exit with 1
#   retained ownership token(s)
#
# The same two rebinds in a shorter function are fine, and so is either one on
# its own -- what this needs is the whole body. Written as calls that RETURN the
# answer (`_marker_start` / `_marker_end` in runtime/lib/traceback.py) it
# compiles, because the value is then the frame's own from the start; that is
# why the stdlib module is spelled that way.
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
