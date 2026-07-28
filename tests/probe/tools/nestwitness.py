#!/usr/bin/env python3
"""Build a minimal program for each empty cell nestgrid.py names, and run it.

nestgrid.py answers "which syntactic nestings has nobody written down". That is
a claim about the corpus, not about the compiler. This answers the next
question: of the nestings nobody wrote down, which ones does today's `lyc` get
WRONG? It synthesises one small program per cell, checks the program is even
legal Python, then runs it against CPython.

    python3 tests/probe/tools/nestwitness.py ./build/bin/lyc \
        tests/golden/cases --reps 5 --out /tmp/w.json
    python3 tests/probe/tools/nestwitness.py ./build/bin/lyc \
        tests/golden/cases --emit try.body:for       # print one witness

FIVE BUCKETS, not four. The four the facts tables use are

    .  stdout and exit status both match CPython
    W  silently wrong: both exited 0 and stdout differs, or lyc completed
       where CPython raised
    R  refused at compile time (a diagnostic, no signal)
    X  aborted or died on a signal

plus a fifth that is NOT a result:

    T  timed out -- UNMEASURED. This box is shared with sibling agents and a
       contended machine can manufacture a timeout that a quiet one does not
       show. rfc/stdlib-semantics.md 13c: contention can fabricate a failure
       but not a pass, so `T` may never be reported as a defect, and a cell
       whose reps disagree is reported as such rather than reduced to a mode.

THE WITNESSES ARE DELIBERATELY DATAFLOW-COUPLED. A syntactic nesting alone does
not reach the shipped SIGSEGV: its minimisation needed a local WRITTEN in the
loop body and READ in the handler, and a witness that merely printed inside the
loop would come back clean and be relayed as "this cell is safe". So every
inner template accumulates into a name the enclosing region also reads, and
every `try` template's handler reads it back. This buys one dataflow shape out
of many; a `.` here means "this cell with THIS coupling is fine", never "this
cell is fine".

THREE WAYS IT REFUSES (rfc/stdlib-semantics.md 13h -- an instrument's false
answer is read as a regression in the thing being measured):

  1. CPython is the oracle, so a witness CPython cannot run has no oracle.
     Those are reported CPYERR with CPython's own stderr and are never
     classified. A witness is the tool's own output, so a bad one is the tool's
     defect, not the compiler's.
  2. A cell whose witness does not COMPILE is IMPOSSIBLE -- the nesting cannot
     be written in Python at all, so its emptiness in the corpus is not a gap.
     The SyntaxError is printed rather than summarised.
  3. A cell this tool cannot spell is NO-WITNESS, which is separate from
     IMPOSSIBLE on purpose. "I could not construct a reaching program" and "no
     reaching program exists" are different claims and only the second is about
     the language. Where the reason IS the grammar -- a statement-only inner in
     an expression-only outer region such as a lambda body -- that reason is
     printed with it.

THE TEMPLATE SET IS NOT EXHAUSTIVE, in the same way nestgrid's axes are not:
one spelling per cell, one dataflow coupling, no `if`, no `match`, no
decorators, no annotations beyond the wrapper's. Every count it produces is a
LOWER BOUND on what is broken.

AND THE SPELLING CAN DECIDE THE VERDICT FOR A WHOLE ROW. Measured, not feared:
the eleven `while.else > *` cells came back `W` and the eleven `for.else > *`
cells came back clean, which reads as a difference between `while` and `for`.
It is not. The `for.else` template happens to write the accumulator in the LOOP
BODY as well as in the `else` (`for i in [1, 2]: acc += i`), while the
`while.else` template writes only the loop counter there -- and
`tests/probe/wb_loopelse_only_write_lost.py` measures that as exactly the
condition separating correct from silently wrong. Respelled to match, the
`for.else` row is wrong too. So a `.` from this tool is a statement about one
program, and a row of them is not a statement about a region.
"""

import argparse
import ast
import json
import pathlib
import subprocess
import sys
import tempfile
from collections import Counter

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import nestgrid  # noqa: E402

CPY = "/opt/homebrew/Frameworks/Python.framework/Versions/3.14/bin/python3.14"

# ------------------------------------------------------------------ templates
#
# `{B}` marks where the nested construct goes; its column decides the indent.
# `V` is the accumulator the region reads back, so the witness carries a
# dataflow edge across the nesting rather than only a syntactic one.

STMT_OUTER = {
    "try.body":      "try:\n    {B}\nexcept ZeroDivisionError:\n    V += 100",
    "try.handler":   "try:\n    raise ZeroDivisionError\n"
                     "except ZeroDivisionError:\n    {B}",
    "try.else":      "try:\n    V += 1\nexcept ZeroDivisionError:\n"
                     "    V += 100\nelse:\n    {B}",
    "try.finally":   "try:\n    V += 1\nfinally:\n    {B}",
    "try*.body":     "try:\n    {B}\nexcept* ZeroDivisionError:\n    V += 100",
    "try*.handler":  "try:\n    raise ExceptionGroup(\"g\", "
                     "[ZeroDivisionError()])\n"
                     "except* ZeroDivisionError:\n    {B}",
    "try*.else":     "try:\n    V += 1\nexcept* ZeroDivisionError:\n"
                     "    V += 100\nelse:\n    {B}",
    "try*.finally":  "try:\n    V += 1\nexcept* ZeroDivisionError:\n"
                     "    V += 100\nfinally:\n    {B}",
    "for.body":      "for i in [1, 2]:\n    {B}",
    "for.else":      "for i in [1, 2]:\n    V += i\nelse:\n    {B}",
    "while.body":    "n = 0\nwhile n < 2:\n    n += 1\n    {B}",
    "while.else":    "n = 0\nwhile n < 2:\n    n += 1\nelse:\n    {B}",
    "with.body":     "with _CM() as c:\n    {B}",
    "afor.body":     "async for i in _agen():\n    {B}",
    "afor.else":     "async for i in _agen():\n    V += i\nelse:\n    {B}",
    "awith.body":    "async with _ACM() as c:\n    {B}",
    # Scope-opening regions. Each declares its OWN accumulator so the nested
    # construct still has a name to write and the region still reads it back;
    # a closure write would need `nonlocal` and would be a different cell.
    "def.body":      "def _inner() -> int:\n    V = 0\n    {B}\n"
                     "    return V\nV += _inner()",
    "gen.body":      "def _inner():\n    V = 0\n    yield 0\n    {B}\n"
                     "    yield V\nV += sum(_inner())",
    "adef.body":     "async def _inner() -> int:\n    V = 0\n    {B}\n"
                     "    return V\nV += await _inner()",
    "agen.body":     "async def _inner():\n    V = 0\n    yield 0\n    {B}\n"
                     "    yield V\nasync for _q in _inner():\n    V += _q",
    "class.body":    "class _K:\n    V = 0\n    {B}\nV += _K.V",
}

# Regions that hold an EXPRESSION, not statements.
EXPR_OUTER = {
    "lambda.body":   "_l = lambda: ({B})\nV += int(_l() is not None)",
    "listcomp.elt":  "V += len([({B}) for i in [1, 2]])",
    "setcomp.elt":   "V += len({({B}) for i in [1, 2]})",
    "dictcomp.elt":  "V += len({i: ({B}) for i in [1, 2]})",
    "genexp.elt":    "V += sum(1 for i in [1, 2] if (({B}) is not None))",
    "listcomp.if":   "V += len([i for i in [1, 2] if (({B}) is not None)])",
    "setcomp.if":    "V += len({i for i in [1, 2] if (({B}) is not None)})",
    "dictcomp.if":   "V += len({i: i for i in [1, 2] if (({B}) is not None)})",
    "genexp.if":     "V += sum(1 for i in [1, 2] if (({B}) is not None))",
    "listcomp.iter": "V += len([i for i in [1, 2] for k in ({B})])",
    "setcomp.iter":  "V += len({i for i in [1, 2] for k in ({B})})",
    "dictcomp.iter": "V += len({i: k for i in [1, 2] for k in ({B})})",
    "genexp.iter":   "V += sum(1 for i in [1, 2] for k in ({B}))",
}

STMT_INNER = {
    "for":         "for j in [1, 2]:\n    V += j",
    "while":       "m = 0\nwhile m < 2:\n    m += 1\n    V += m",
    "try":         "try:\n    V += 1\nexcept ZeroDivisionError:\n    V += 100",
    "try*":        "try:\n    V += 1\nexcept* ZeroDivisionError:\n    V += 100",
    "try/finally": "try:\n    V += 1\nfinally:\n    V += 2",
    "with":        "with _CM() as c:\n    V += c",
    "afor":        "async for q in _agen():\n    V += q",
    "awith":       "async with _ACM() as c:\n    V += c",
    "def":         "def _f() -> int:\n    return 3\nV += _f()",
    "gen":         "def _g():\n    yield 3\nV += sum(_g())",
    "adef":        "async def _af() -> int:\n    return 3\nV += await _af()",
    "agen":        "async def _ag():\n    yield 3\nasync for _z in _ag():\n"
                   "    V += _z",
    "class":       "class _C:\n    def m(self) -> int:\n        return 3\n"
                   "V += _C().m()",
    "raise":       "raise ZeroDivisionError",
    "return":      "return V + 7",
    "break":       "V += 1\nbreak",
    "continue":    "V += 1\ncontinue",
    "yield":       "V += 1\nyield V",
    "lambda":      "_lm = lambda: 3\nV += _lm()",
    "listcomp":    "V += sum([j for j in [1, 2]])",
    "setcomp":     "V += len({j for j in [1, 2]})",
    "dictcomp":    "V += len({j: j for j in [1, 2]})",
    "genexp":      "V += sum(j for j in [1, 2])",
    "await":       "V += await _af2()",
}

EXPR_INNER = {
    "lambda":   "(lambda: 3)()",
    "listcomp": "[j for j in [1, 2]]",
    "setcomp":  "{j for j in [1, 2]}",
    "dictcomp": "{j: j for j in [1, 2]}",
    "genexp":   "sum(j for j in [1, 2])",
    "yield":    "(yield 1)",
    "await":    "(await _af2())",
}

# Inner kinds that have no expression spelling in Python at all. Reported with
# their reason rather than as a limitation of this tool.
STATEMENT_ONLY = {"for", "while", "try", "try*", "try/finally", "with", "afor",
                  "awith", "def", "gen", "adef", "agen", "class", "raise",
                  "return", "break", "continue"}

ASYNC_KINDS = {"afor", "awith", "adef", "agen", "await"}
ASYNC_REGIONS = {"afor.body", "afor.else", "awith.body", "adef.body",
                 "agen.body"}

PRELUDE_CM = ("class _CM:\n"
              "    def __enter__(self) -> int:\n        return 5\n"
              "    def __exit__(self, a, b, c) -> bool:\n        return False\n")
PRELUDE_ACM = ("class _ACM:\n"
               "    async def __aenter__(self) -> int:\n        return 5\n"
               "    async def __aexit__(self, a, b, c) -> bool:\n"
               "        return False\n")
PRELUDE_AGEN = "async def _agen():\n    yield 1\n    yield 2\n"
PRELUDE_AF2 = "async def _af2() -> int:\n    return 3\n"


def splice(template: str, body: str) -> str:
    """Put `body` where `{B}` is, indented to `{B}`'s own column."""
    out = []
    for line in template.split("\n"):
        if "{B}" not in line:
            out.append(line)
            continue
        pad = line[:len(line) - len(line.lstrip())]
        head = line.split("{B}")[0].strip()
        if head:  # `{B}` is mid-line: an expression slot.
            out.append(line.replace("{B}", body))
            continue
        out.extend(pad + b if b.strip() else b for b in body.split("\n"))
    return "\n".join(out)


def build(outer: str, inner: str) -> "tuple[str | None, str]":
    """(source, note). source is None when no witness can be spelled."""
    expr_ctx = outer in EXPR_OUTER
    if expr_ctx:
        if inner in STATEMENT_ONLY:
            return None, (f"{inner} is statement-only and {outer} is an "
                          f"expression position: impossible by grammar")
        if inner not in EXPR_INNER:
            return None, f"no expression spelling for {inner} in this tool"
        body = EXPR_INNER[inner]
        template = EXPR_OUTER[outer]
    else:
        if outer not in STMT_OUTER:
            return None, f"no template for region {outer}"
        if inner not in STMT_INNER:
            return None, f"no statement spelling for {inner} in this tool"
        body = STMT_INNER[inner]
        template = STMT_OUTER[outer]

    block = splice(template, body)

    # `break`/`continue` need a loop the cell does not itself provide; `yield`
    # needs a function, which the wrapper always is. Adding the loop OUTSIDE
    # the tested region leaves the (outer, inner) relation untouched.
    if inner in ("break", "continue") and outer not in (
            "for.body", "for.else", "while.body", "while.else", "afor.body"):
        block = splice("for _b in [1]:\n    {B}", block)
    # A bare `raise` has to be caught somewhere or the witness exits nonzero on
    # BOTH sides, which is a comparison with no information in it.
    if inner == "raise" and outer not in ("try.body", "try*.body"):
        block = splice("try:\n    {B}\nexcept ZeroDivisionError:\n    V += 5",
                       block)

    if (outer, inner) in nestgrid.AXIS_IMPOSSIBLE:
        return None, ("empty by this tool's own axis split, not by the "
                      "language: nestgrid puts a yielding def in gen.body")

    is_async = (inner in ASYNC_KINDS or outer in ASYNC_REGIONS)
    # A `yield` in the outermost region turns the WRAPPER into a generator, so
    # it can no longer return `acc` and has to be drained instead. The regions
    # excluded here supply their own inner function, which absorbs the yield.
    yields = (inner == "yield" and outer not in
              ("gen.body", "agen.body", "def.body", "adef.body", "class.body")
              and not expr_ctx)

    prelude = ""
    text = block
    if "_CM(" in text:
        prelude += PRELUDE_CM
    if "_ACM(" in text:
        prelude += PRELUDE_ACM
    if "_agen(" in text:
        prelude += PRELUDE_AGEN
    if "_af2(" in text:
        prelude += PRELUDE_AF2

    kw = "async def" if is_async else "def"
    src = [prelude] if prelude else []
    src.append(f"{kw} w():\n    acc = 0\n"
               + splice("    {B}", block).rstrip("\n"))
    if yields and is_async:
        # An async function that yields is an ASYNC generator: `sum()` cannot
        # drain it and downgrading the wrapper to a plain `def` would delete
        # the `async for`/`async with` the cell is about. Drain it with the
        # only construct that can.
        src.insert(0, "import asyncio")
        src.append("async def _drive():\n    t = 0\n    async for _v in w():\n"
                   "        t += _v\n    return t\nprint(asyncio.run(_drive()))")
    elif yields:
        src.append("print(sum(w()))")
    else:
        src[-1] += "\n    return acc"
        if is_async:
            src.insert(0, "import asyncio")
            src.append("print(asyncio.run(w()))")
        else:
            src.append("print(w())")
    return "\n".join(src).replace("V", "acc") + "\n", ""


def run(cmd, timeout):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=timeout, stdin=subprocess.DEVNULL)
        return r.returncode, r.stdout, r.stderr
    except subprocess.TimeoutExpired:
        return None, "", ""


def bucket(lyc_rc, lyc_out, ref_rc, ref_out):
    if lyc_rc is None:
        return "T"
    if lyc_rc < 0 or lyc_rc in (134, 138, 139):
        return "X"
    if lyc_rc == ref_rc and lyc_out == ref_out:
        return "."
    if lyc_rc == 0:
        # Completed where CPython raised, or produced a different answer. Both
        # are the silent direction: nothing tells the caller anything is wrong.
        return "W"
    return "R"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("lyc", type=pathlib.Path)
    ap.add_argument("corpus", type=pathlib.Path)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--emit", metavar="OUTER:INNER",
                    help="print one witness and exit; nothing is run")
    ap.add_argument("--cells", nargs="*", default=None,
                    help="OUTER:INNER pairs to test instead of the corpus gaps")
    ap.add_argument("--out", type=pathlib.Path)
    ap.add_argument("--allow-unparsed", action="append", default=[])
    args = ap.parse_args()

    if args.emit:
        outer, inner = args.emit.split(":")
        src, note = build(outer, inner)
        print(src if src else f"NO-WITNESS: {note}")
        return 0 if src else 2

    if args.cells:
        cells = [tuple(c.split(":")) for c in args.cells]
        print(f"cells: {len(cells)} given on the command line")
    else:
        res = nestgrid.scan_corpus(args.corpus)
        bad = [p for p, _ in res["unparsed"] if p.name not in args.allow_unparsed]
        if bad:
            print(f"REFUSED: {len(bad)} file(s) in {args.corpus} did not parse; "
                  f"the gap list would be computed over a silently smaller "
                  f"corpus: {[str(p) for p in bad]}", file=sys.stderr)
            return 2
        cells = []
        for outer in nestgrid.OUTER_ORDER:
            for inner in nestgrid.INNER_ORDER:
                cell = (outer, inner)
                if res["cells_frame"][cell]:
                    continue
                if nestgrid.classify_zero(outer, inner, res) != "GAP" \
                        and not res["cells"][cell]:
                    continue
                cells.append(cell)
        print(f"corpus: {args.corpus} -- "
              f"{len(res['files']) - len(res['unparsed'])} files, "
              f"{res['nodes']} nodes")
        print(f"cells with no in-frame coverage and both features present: "
              f"{len(cells)} of {len(nestgrid.OUTER_ORDER) * len(nestgrid.INNER_ORDER)}")

    rows = []
    tally = Counter()
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="nestwitness."))
    for outer, inner in cells:
        name = f"{outer}:{inner}"
        src, note = build(outer, inner)
        if src is None:
            tally["NO-WITNESS"] += 1
            rows.append(dict(cell=name, verdict="NO-WITNESS", note=note))
            print(f"  NO-WITNESS  {name:<28} {note}", flush=True)
            continue
        try:
            compile(src, name, "exec")
        except SyntaxError as exc:
            tally["IMPOSSIBLE"] += 1
            rows.append(dict(cell=name, verdict="IMPOSSIBLE", note=str(exc),
                             src=src))
            print(f"  IMPOSSIBLE  {name:<28} {exc}", flush=True)
            continue
        # `try/finally` and `try*.body` are cell names, not path components.
        safe = name
        for ch, rep in ((".", "_"), (":", "__"), ("/", "_"), ("*", "s")):
            safe = safe.replace(ch, rep)
        path = tmp / (safe + ".py")
        path.write_text(src)
        ref_rc, ref_out, ref_err = run([CPY, str(path)], args.timeout)
        if ref_rc is None or ref_rc != 0:
            tally["CPYERR"] += 1
            rows.append(dict(cell=name, verdict="CPYERR", src=src,
                             note=ref_err.strip().split("\n")[-1] if ref_err
                             else "timeout"))
            print(f"  CPYERR      {name:<28} "
                  f"{rows[-1]['note']}  <- the WITNESS is wrong, not lyc",
                  flush=True)
            continue
        seen = "".join(bucket(*run([str(args.lyc), "jit", str(path)],
                                   args.timeout)[:2], ref_rc, ref_out)
                       for _ in range(args.reps))
        verdict = "MIXED" if len(set(seen)) > 1 else seen[0]
        tally[verdict] += 1
        rows.append(dict(cell=name, verdict=verdict, reps=seen, src=src,
                         expected=ref_out.strip()))
        print(f"  {verdict:<11} {name:<28} {seen}  cpython={ref_out.strip()!r}",
              flush=True)
        # Written after every cell, not at the end: a sweep is ~90 minutes and
        # this tool has already lost one by dying on cell 43. A partial file
        # that says how far it got beats a complete one that never appears.
        if args.out:
            args.out.write_text(json.dumps(rows, indent=1))

    print(f"\ntally over {len(cells)} cells: "
          + "  ".join(f"{k}={v}" for k, v in sorted(tally.items())))
    print("`.`=matches CPython  W=silent wrong  R=refused  X=abort/signal  "
          "T=TIMED OUT, unmeasured, never a defect  MIXED=reps disagreed")
    print(f"witnesses kept in {tmp}")
    if args.out:
        args.out.write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
