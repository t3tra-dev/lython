"""Runtime objects for the exception traceback (CPython's Objects/frameobject.c,
Objects/codeobject.c and Python/traceback.c counterparts).

The rest of `types` is a contract manifest (runtime/modules/types.mlir); these
three are here because they are the only members of it that a program HOLDS --
`e.__traceback__` hands one back, and `traceback.py` walks it. A manifest
`py.class` is a typing contract with no layout, so a value of one cannot exist;
written here they are ordinary classes with a synthesized layout, allocation and
deallocator, and `tb_next: Optional["TracebackType"]` is a boxed field like any
other self-referential link.

Deviations from CPython:
- `CodeType` carries `co_filename`, `co_name` and `co_firstlineno` only. The
  rest of CPython's code object describes BYTECODE (co_code, co_consts,
  co_varnames, co_stacksize, ...), which a compiled program does not have.
- `FrameType` carries `f_code` and `f_lineno` only. `f_back`, `f_globals`,
  `f_locals` and `f_trace` are the interpreter's frame chain and namespaces;
  the traceback's frames are recorded at raise sites, so there is no live frame
  object behind them to expose.
- `tb_lasti` is always -1. It is a bytecode offset, and there is none; the
  column range it would index into `co_positions()` is carried by the link
  itself instead.
- None of the three is constructible from CPython in the general case either
  (`CodeType` needs 18 arguments); they are written with `__init__` here
  because the traceback builder is Python.
"""

from typing import Optional


class CodeType:
    """The static half of a traceback frame: which file, which callable."""

    co_filename: str
    co_name: str
    co_firstlineno: int

    def __init__(self, co_filename: str, co_name: str,
                 co_firstlineno: int) -> None:
        self.co_filename = co_filename
        self.co_name = co_name
        self.co_firstlineno = co_firstlineno


class FrameType:
    """One recorded frame. `f_lineno` is the line the raise passed through."""

    f_code: CodeType
    f_lineno: int

    def __init__(self, f_code: CodeType, f_lineno: int) -> None:
        self.f_code = f_code
        self.f_lineno = f_lineno


class TracebackType:
    """A link of the traceback chain, oldest frame first (CPython's order).

    ⛔ `tb_colno` / `tb_end_colno` / `tb_marker` ARE THIS LINK'S OWN, and CPython
    has no such attributes. It reaches the same numbers through
    `tb_frame.f_code.co_positions()[tb_lasti]` -- the code object's position
    table, indexed by the bytecode offset that failed. There is neither here, and
    a recorded frame IS one position, so the link carries it. `tb_marker` is the
    anchor mode the runtime recorded: 0 draws nothing, 1 applies the call /
    operator heuristics, 2 renders the plain range.
    """

    tb_next: Optional["TracebackType"]
    tb_frame: FrameType
    tb_lasti: int
    tb_lineno: int
    tb_colno: int
    tb_end_colno: int
    tb_marker: int

    def __init__(self, tb_next: Optional["TracebackType"], tb_frame: FrameType,
                 tb_lasti: int, tb_lineno: int, tb_colno: int = 0,
                 tb_end_colno: int = 0, tb_marker: int = 0) -> None:
        self.tb_next = tb_next
        self.tb_frame = tb_frame
        self.tb_lasti = tb_lasti
        self.tb_lineno = tb_lineno
        self.tb_colno = tb_colno
        self.tb_end_colno = tb_end_colno
        self.tb_marker = tb_marker
