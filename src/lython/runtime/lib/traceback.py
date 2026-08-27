"""Formatting and printing of exception tracebacks (CPython's Lib/traceback.py).

WHERE THE FRAMES COME FROM. CPython's traceback objects are built by the
interpreter as it unwinds; here the runtime records a frame at every raise site
it passes through (lowering/Common/TracebackSupportBuilder.cpp), and the native
half of this module -- `_traceback` -- reads that stack back. `_current_tb()`
turns it into the `types.TracebackType` chain the public functions walk, so a
caught traceback and the one an uncaught exception prints come from one place
and cannot disagree.

Deviations from CPython:
- `print_exc`, `format_exc` and `print_exception` describe the exception being
  HANDLED. The frame stack is the in-flight exception's and is cleared once it
  is handled, so these say something only inside an `except` block; outside one
  `format_exc()` answers 'NoneType: None\n', as CPython's does.
- there is no `sys.exc_info()`, so `print_exception(exc)` takes the exception
  itself (CPython 3.10+ accepts that form too) and never the legacy
  (type, value, tb) triple. `limit` is keyword-only there, as it is in CPython's
  modern form, because the second positional is the legacy `value`.
- `extract_stack`, `format_stack`, `print_stack` and `walk_stack` are not
  provided: they report the CALL stack, and a compiled program keeps no frame
  objects for frames that are not raising.
- `TracebackException`, `StackSummary.from_list`, `chain=`, `__cause__` and
  `__context__` display are not provided. The chain is recorded and the
  uncaught printer walks it; reaching it as a VALUE needs `e.__cause__`, which
  is still refused.
- `FrameSummary` carries `filename`, `lineno`, `name`, `line` and the column
  range CPython's `colno` / `end_colno` carry. It does not carry `locals`.
  `line` is stripped and `_original_line` is not, as in CPython, but both are
  plain fields rather than properties over a `_lines` pair.
- WHETHER a frame gets a `~~~^^^` line is decided at COMPILE time, not here.
  CPython's `_should_show_carets` parses the source line back to ask whether the
  statement is `return f(...)` or `x = f(...)`, which it draws nothing under;
  `ast` is not reachable from a compiled program, and the emitter has the tree
  anyway, so it answers there and the answer arrives as `tb_marker`. WHERE the
  carets go is decided here, by `_anchors`, from the text.
- `StackSummary` is a class holding a list rather than a `list` subclass.

⛔ THE ANNOTATIONS NAME `types.TracebackType`, NOT A RE-EXPORTED `TracebackType`.
`from types import TracebackType` binds the name here, and a return annotation
written with it reaches a caller as this module's name for the class rather than
as the class -- reading a field off the result then fails with "attr.get object
type has no class schema", which names nothing the caller wrote. Importing the
module and qualifying is what carries the contract across.
- source lines are read with one `open()` per frame. CPython's `linecache`
  caches whole files; a traceback is printed once, so the cache would outlive
  its use.
- a frame whose file is gone prints without its source line, as CPython's does.
"""

from typing import Optional

import _traceback
import os
import sys
import types


def _read_source_line(filename: str, lineno: int) -> str:
    """The `lineno`-th line of `filename` with its trailing newline removed, or
    '' if unreadable. The LEADING whitespace stays: the recorded columns are
    absolute, so the anchor line needs to know how much the display strips.

    ⛔ The file is checked for BEFORE it is opened rather than opened inside a
    `try`: a name bound in a try body does not escape the statement here, so the
    handle would not be in scope to read or close.
    """
    if lineno <= 0:
        return ""
    if not os.path.exists(filename):
        return ""
    handle = open(filename, "r")
    found = ""
    n = 0
    while True:
        line = handle.readline()
        if line == "":
            break
        n += 1
        if n == lineno:
            found = line
            break
    handle.close()
    return found.rstrip()


def _current_tb() -> Optional[types.TracebackType]:
    """The traceback of the exception being handled, or None outside a handler.

    The recorded stack runs innermost first; a traceback chain runs outermost
    first, so building it by prepending each frame in stack order lands the
    outermost one at the head.
    """
    count = _traceback.frame_count()
    built: Optional[types.TracebackType] = None
    i = 0
    while i < count:
        line = _traceback.frame_line(i)
        code = types.CodeType(_traceback.frame_file(i),
                              _traceback.frame_name(i), line)
        built = types.TracebackType(built, types.FrameType(code, line), -1,
                                    line, _traceback.frame_col(i),
                                    _traceback.frame_end_col(i),
                                    _traceback.frame_marker(i))
        i += 1
    return built


def _is_closer(ch: str) -> bool:
    return ch == ")" or ch == "]"


def _is_operator_char(ch: str) -> bool:
    return (ch == "+" or ch == "-" or ch == "*" or ch == "/" or ch == "%"
            or ch == "@" or ch == "&" or ch == "|" or ch == "^" or ch == "<"
            or ch == ">")


def _anchors(line: str, col: int, end_col: int, mode: int) -> str:
    """CPython's `~~~^^^` underline for the failing range of `line`.

    `line` is the DISPLAYED line -- the source with its indentation removed --
    and the columns have been shifted by that indentation, because the recorded
    ones are absolute. Returns '' where CPython draws nothing.

    ⭐ THE SAME HEURISTICS THE UNCAUGHT PRINTER USES, and deliberately the same
    ones: `print_marker` in lowering/Common/TracebackSupportBuilder.cpp reads
    this exact stack, and a traceback a program formats must not be able to
    disagree with one the runtime prints. CPython derives the anchors from the
    instruction that failed; there is no instruction here, so both read the
    source text -- a call or subscript range splits at its first `(` or `[` and
    renders `~~~^^^`, an operator range puts the carets over the operator run,
    and a range with neither renders all carets unless it covers the whole line,
    which is CPython's last `_should_show_carets` test written without an AST:
    nothing before the range and nothing after it means the underline would say
    only "all of it".

    Whether a frame reaches this at all was already decided by the compiler; see
    the module docstring.
    """
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
        # An operator OUTSIDE every bracket is the segment's top node, so it
        # takes the carets: `f(x) + g(y)` is an addition, not a call.
        op = start
        depth = 0
        while op < marker_end:
            ch = line[op]
            if ch == "(" or ch == "[":
                depth += 1
            elif ch == ")" or ch == "]":
                depth -= 1
            elif depth == 0 and _is_operator_char(ch):
                run = op + 1
                while run < marker_end and _is_operator_char(line[run]):
                    run += 1
                caret_start = op
                caret_end = run
                break
            op += 1
        if caret_start < 0 and _is_closer(line[marker_end - 1]):
            # With no operator above them, the LAST bracket group is the call
            # or subscript being made, and its opener is where the carets
            # start. Found by matching backwards from the closer rather than
            # forwards from `start`, which answered `Box(3).bad()` with the
            # arguments of `Box`.
            scan = marker_end - 1
            depth = 0
            while scan >= start:
                ch = line[scan]
                if ch == ")" or ch == "]":
                    depth += 1
                elif ch == "(" or ch == "[":
                    depth -= 1
                    if depth == 0:
                        caret_start = scan
                        caret_end = marker_end
                        break
                scan -= 1
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


class FrameSummary:
    """One line of a traceback: where it was and what the source says."""

    filename: str
    lineno: int
    name: str
    line: str
    # The source line with its indentation still on, CPython's
    # `_original_line`. `line` is what a reader wants and the anchors are
    # measured against columns that count from the start of the real line, so
    # the amount `line` dropped has to stay reachable.
    _original_line: str
    colno: int
    end_colno: int
    marker: int

    def __init__(self, filename: str, lineno: int, name: str, line: str,
                 colno: int, end_colno: int, marker: int) -> None:
        self.filename = filename
        self.lineno = lineno
        self.name = name
        self.line = line.strip()
        self._original_line = line
        self.colno = colno
        self.end_colno = end_colno
        self.marker = marker

    def __repr__(self) -> str:
        return ("<FrameSummary file " + self.filename + ", line "
                + str(self.lineno) + " in " + self.name + ">")


class StackSummary:
    """A list of FrameSummary, oldest frame first."""

    frames: list[FrameSummary]

    def __init__(self, frames: list[FrameSummary]) -> None:
        self.frames = frames

    def format(self) -> list[str]:
        out: list[str] = []
        for frame in self.frames:
            text = ('  File "' + frame.filename + '", line '
                    + str(frame.lineno) + ", in " + frame.name + "\n")
            if frame.line != "":
                shown = frame.line
                text += "    " + shown + "\n"
                if frame.marker != 0:
                    original = frame._original_line
                    indent = len(original) - len(original.lstrip())
                    col = frame.colno - indent
                    end = frame.end_colno - indent
                    if col < 0:
                        col = 0
                    if end < 0:
                        end = 0
                    anchors = _anchors(shown, col, end, frame.marker)
                    if anchors != "":
                        text += "    " + anchors + "\n"
            out.append(text)
        return out


def extract_tb(tb: Optional[types.TracebackType],
               limit: Optional[int] = None) -> StackSummary:
    """The frames of `tb` as a StackSummary, oldest first."""
    frames: list[FrameSummary] = []
    cur = tb
    taken = 0
    while cur is not None:
        if limit is not None and taken >= limit:
            break
        code = cur.tb_frame.f_code
        frames.append(FrameSummary(code.co_filename, cur.tb_lineno,
                                   code.co_name,
                                   _read_source_line(code.co_filename,
                                                     cur.tb_lineno),
                                   cur.tb_colno, cur.tb_end_colno,
                                   cur.tb_marker))
        taken += 1
        cur = cur.tb_next
    return StackSummary(frames)


def format_list(extracted: StackSummary) -> list[str]:
    """One string per frame, each ending in a newline."""
    return extracted.format()


def format_tb(tb: Optional[types.TracebackType],
              limit: Optional[int] = None) -> list[str]:
    """`extract_tb` formatted."""
    return extract_tb(tb, limit).format()


def print_tb(tb: Optional[types.TracebackType],
             limit: Optional[int] = None) -> None:
    """Write `format_tb` to stderr."""
    for text in format_tb(tb, limit):
        sys.stderr.write(text)


def format_exception_only(exc: BaseException) -> list[str]:
    """The last line of a traceback: the class and, when it has one, the str."""
    label = type(exc).__name__
    message = str(exc)
    if message == "":
        return [label + "\n"]
    return [label + ": " + message + "\n"]


def format_exception(exc: BaseException, *,
                     limit: Optional[int] = None) -> list[str]:
    """The whole traceback of the exception being handled, as CPython lays it
    out: the header, one entry per frame, then the exception line."""
    out: list[str] = []
    frames = format_tb(_current_tb(), limit)
    if len(frames) > 0:
        out.append("Traceback (most recent call last):\n")
        for text in frames:
            out.append(text)
    for text in format_exception_only(exc):
        out.append(text)
    return out


def print_exception(exc: BaseException, *,
                    limit: Optional[int] = None) -> None:
    """Write `format_exception` to stderr."""
    for text in format_exception(exc, limit=limit):
        sys.stderr.write(text)


def format_exc(limit: Optional[int] = None) -> str:
    """The traceback of the exception being handled, as one string.

    Outside a handler CPython answers 'NoneType: None\n', which is what
    formatting a `None` exception produces; there is nothing in flight to
    describe, so the same string is what this says.
    """
    line = _traceback.exc_line()
    if line == "":
        return "NoneType: None\n"
    out: list[str] = []
    frames = format_tb(_current_tb(), limit)
    if len(frames) > 0:
        out.append("Traceback (most recent call last):\n")
        for text in frames:
            out.append(text)
    out.append(line + "\n")
    return "".join(out)


def print_exc(limit: Optional[int] = None) -> None:
    """Write `format_exc` to stderr."""
    sys.stderr.write(format_exc(limit))
