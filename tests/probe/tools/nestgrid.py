#!/usr/bin/env python3
"""Which SYNTACTIC NESTINGS does the corpus never exercise? Name the empty cells.

A shipped SIGSEGV survived 490 tests:

    try:
        for v in [1, 2]:      # no generator, no exception ever raised
            total += v
    except ZeroDivisionError:
        total += 100

`rc=139`, deterministic, CPython prints 3. Of 291 golden cases, 88 contain
`try` and 86 contain `for` -- and ZERO put the `for` in the `try` BODY. The
defect is not rare. The test is rare.

The existing duplication census (83,028 pairs, 20 identical) measures how
similar two PROGRAMS are. Nothing measured whether a syntactic COMBINATION had
ever been written down. This does.

WHY THE DIRECTION OF NESTING IS THE WHOLE POINT. "contains `try`" and "has a
`for` INSIDE a `try` body" are different facts and only the second was zero.
So a cell here is an ordered pair (outer REGION, inner kind), where the region
names the field descended through, not just the node:

    try.body > for          the shipped crash
    try.handler > for       15 files have this; the crash needs the other one

WHY NOT REGEX, and why not `ast.walk`. Both were tried by the coordinator and
both produced a false positive on the same file: `runtime_tuple_iteration` was
reported as `for` inside `try`, but the `for` was in an `except*` HANDLER.
`ast.walk` flattens the tree and cannot tell body from handler at all; a regex
cannot tell them apart without reimplementing the indentation rules. This walks
parent-to-child so the region label is a byproduct of the descent rather than a
thing to be inferred afterwards.

WHY SCOPE BOUNDARIES ARE TRACKED SEPARATELY. The mechanism behind the crash is
per-function: unwind pads, cell allocations and loop-carried locals all live in
one function body. A `for` reached from a `try` only by descending into a nested
`def` is a different fact from one in the same frame, so containment is reported
as `same-scope` (no def/lambda/class/comprehension boundary crossed) with the
cross-scope total beside it. Comprehension `iter` of the FIRST generator is
evaluated in the ENCLOSING scope -- Python evaluates it before the implicit
function is called -- so that one edge does not count as a crossing.

REFUSALS. Three, because a coverage grid full of zeros is exactly what a broken
scanner also produces (rfc/stdlib-semantics.md 13g: six instruments all
self-reported "nothing unusual"):

  1. A file that does not parse ABORTS the run. Skipping shrinks the
     denominator silently, and Lython accepts CPython 3.14 syntax, so the host
     interpreter's grammar is a real variable -- its version is printed and any
     rejected file is named, never counted as "no findings here".
  2. Files and nodes scanned are printed BEFORE any result, because an empty
     cell reads identically as "never written" and "never looked at".
  3. Empty cells are split three ways. A cell whose OUTER region never occurs
     anywhere in the corpus, or whose INNER kind never occurs, is vacuous -- it
     is not a gap, and listing it as one buries the real ones. Only cells where
     BOTH features occur separately but never together are reported as GAP.

THE COUPLED COLUMN, and the wrong guess that produced it. The coordinator
measured `try.body > for` as 0 over 291 golden cases. This tool measured it as
ONE, in `dict_iteration_views.py`, added by fcc81ca after that snapshot -- and
the SIGSEGV still ships. So syntax alone does not decide, and the first guess
here was that the missing condition is a function FRAME, since the cell
allocation and the unwind pad are per-function and that file's pair is at
module level.

THAT GUESS WAS WRONG, and an ablation says so: the same program at module level
crashes 3/3, and the same program in a function does NOT crash when the handler
reads nothing the loop wrote. What decides is the DATAFLOW EDGE that
`wb_forloop_handler_local_unwind.py` already named as condition (c) -- a local
written inside the nested construct and read again in a SIBLING region of the
same statement. `dict_iteration_views.py` is green because its handler prints
the caught exception, not the loop's accumulator.

So the second count each cell carries is that edge: a name stored inside the
inner construct and loaded in another region of the same outer statement
(`try` body vs its handler, `else` or `finally`; loop body vs its `else`).
`try.body > for` is 1 file syntactically and 0 files COUPLED, and the coupled
number is the one the crash tracks. AugAssign targets count as both stored and
loaded, because `acc += v` reads `acc` while `ast` marks the target Store.

A count going from 0 to 1 without the defect moving is the sharpest available
statement of the caveat below: nonzero is not coverage. The frame distinction
is still printed, because it is real and cheap, but it is NOT the
discriminator and no reading of this grid should treat it as one.

THE AXIS LIST IS NOT EXHAUSTIVE, so every count here is a LOWER BOUND on the
gaps. It covers the statement-level compound forms plus the escape statements
that interact with unwinding. It deliberately omits `if`/`else` (no unwind or
scope semantics of its own), match statements, decorators, `assert`, and every
DATAFLOW property -- and dataflow is load-bearing for the crash above, whose
minimisation needed "a local written in the loop body and READ in the handler".
Two files can land in the same cell here and differ on exactly that. A cell with
a nonzero count is therefore NOT a claim that the shape is covered; only the
zeros are claims, and they are claims about syntax alone.

    python3 tests/probe/tools/nestgrid.py tests/golden/cases tests/golden/errors
    python3 tests/probe/tools/nestgrid.py --gaps-only tests/golden/cases

Exit 0 = scanned cleanly. 2 = refused (a file did not parse, or no input).
"""

import argparse
import ast
import pathlib
import sys
from collections import Counter

# ---------------------------------------------------------------- axes

# Region labels for descending from a node into one of its fields. The key is
# (node class name, field name); ExceptHandler is special-cased because its
# label depends on whether the owning statement is `try` or `try*`, which the
# handler node itself does not record.
REGIONS = {
    ("Try", "body"): "try.body",
    ("Try", "orelse"): "try.else",
    ("Try", "finalbody"): "try.finally",
    ("TryStar", "body"): "try*.body",
    ("TryStar", "orelse"): "try*.else",
    ("TryStar", "finalbody"): "try*.finally",
    ("For", "body"): "for.body",
    ("For", "orelse"): "for.else",
    ("AsyncFor", "body"): "afor.body",
    ("AsyncFor", "orelse"): "afor.else",
    ("While", "body"): "while.body",
    ("While", "orelse"): "while.else",
    ("With", "body"): "with.body",
    ("AsyncWith", "body"): "awith.body",
    ("Lambda", "body"): "lambda.body",
    ("ClassDef", "body"): "class.body",
}

# Regions that open a new scope, so anything already on the stack is reached
# only by crossing a frame boundary.
SCOPE_REGIONS = {
    "def.body", "gen.body", "adef.body", "agen.body",
    "lambda.body", "class.body",
    "listcomp.elt", "setcomp.elt", "dictcomp.elt", "genexp.elt",
    "listcomp.if", "setcomp.if", "dictcomp.if", "genexp.if",
    "listcomp.iter", "setcomp.iter", "dictcomp.iter", "genexp.iter",
}

COMP_KIND = {"ListComp": "listcomp", "SetComp": "setcomp",
             "DictComp": "dictcomp", "GeneratorExp": "genexp"}

# What a node registers AS when found inside a region. A node can register more
# than once: `try:/except:/finally:` is both a `try` and a `try/finally`,
# because the two lower through different machinery.
def inner_kinds(node):
    cls = type(node).__name__
    if cls == "Try":
        out = []
        if node.handlers:
            out.append("try")
        if node.finalbody:
            out.append("try/finally")
        return out
    if cls == "TryStar":
        out = ["try*"]
        if node.finalbody:
            out.append("try/finally")
        return out
    simple = {
        "For": "for", "AsyncFor": "afor", "While": "while",
        "With": "with", "AsyncWith": "awith",
        "ListComp": "listcomp", "SetComp": "setcomp",
        "DictComp": "dictcomp", "GeneratorExp": "genexp",
        "Yield": "yield", "YieldFrom": "yield", "Await": "await",
        "Lambda": "lambda", "ClassDef": "class",
        "Raise": "raise", "Return": "return",
        "Break": "break", "Continue": "continue",
    }
    if cls in simple:
        return [simple[cls]]
    if cls == "FunctionDef":
        return ["gen"] if is_generator(node) else ["def"]
    if cls == "AsyncFunctionDef":
        return ["agen"] if is_generator(node) else ["adef"]
    return []


INNER_ORDER = ["try", "try*", "try/finally", "for", "afor", "while",
               "with", "awith", "listcomp", "setcomp", "dictcomp", "genexp",
               "yield", "await", "def", "gen", "adef", "agen", "lambda",
               "class", "raise", "return", "break", "continue"]

OUTER_ORDER = ["try.body", "try.handler", "try.else", "try.finally",
               "try*.body", "try*.handler", "try*.else", "try*.finally",
               "for.body", "for.else", "while.body", "while.else",
               "afor.body", "afor.else", "with.body", "awith.body",
               "def.body", "gen.body", "adef.body", "agen.body",
               "lambda.body", "class.body",
               "listcomp.elt", "setcomp.elt", "dictcomp.elt", "genexp.elt",
               "listcomp.iter", "setcomp.iter", "dictcomp.iter", "genexp.iter",
               "listcomp.if", "setcomp.if", "dictcomp.if", "genexp.if"]


def is_generator(fn):
    """Does this function's OWN scope yield? Nested defs do not make it one."""
    stack = list(fn.body)
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.Yield, ast.YieldFrom)):
            return True
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.Lambda, ast.ClassDef)):
            continue
        stack.extend(ast.iter_child_nodes(node))
    return False


# ---------------------------------------------------------------- walk

# Sibling regions of one compound statement, for the coupling test. `with` and
# the comprehensions have exactly one region each, so nothing can be coupled
# ACROSS regions there and the column is legitimately always zero for them --
# not a gap, an absence of anywhere for the edge to go.
SIBLING_FIELDS = {
    "Try": ("body", "handlers", "orelse", "finalbody"),
    "TryStar": ("body", "handlers", "orelse", "finalbody"),
    "For": ("body", "orelse"),
    "AsyncFor": ("body", "orelse"),
    "While": ("body", "orelse"),
}


def names(node, want):
    """Names a subtree stores or loads. `want` is "store" or "load".

    AugAssign is counted BOTH ways on purpose: `acc += v` reads acc and writes
    it, but `ast` gives the target ctx=Store only, so asking for loads without
    this would miss every accumulator in the corpus -- which is precisely the
    shape the shipped crash needs.
    """
    out = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name):
            if isinstance(sub.ctx, ast.Store) and want == "store":
                out.add(sub.id)
            elif isinstance(sub.ctx, ast.Load) and want == "load":
                out.add(sub.id)
        elif isinstance(sub, ast.AugAssign) and want == "load":
            for tgt in ast.walk(sub.target):
                if isinstance(tgt, ast.Name):
                    out.add(tgt.id)
    return out


def sibling_loads(owner, field):
    """Names loaded anywhere in `owner` except inside the region `field`."""
    fields = SIBLING_FIELDS.get(type(owner).__name__)
    if not fields:
        return set()
    out = set()
    for other in fields:
        if other == field:
            continue
        for sub in as_nodes(getattr(owner, other, None)):
            out |= names(sub, "load")
    return out


FRAME_REGIONS = {"def.body", "gen.body", "adef.body", "agen.body",
                 "lambda.body",
                 "listcomp.elt", "setcomp.elt", "dictcomp.elt", "genexp.elt",
                 "listcomp.if", "setcomp.if", "dictcomp.if", "genexp.if",
                 "listcomp.iter", "setcomp.iter", "dictcomp.iter",
                 "genexp.iter"}


class Scan:
    def __init__(self):
        self.pairs_same = set()   # (outer, inner) reached without leaving scope
        self.pairs_any = set()
        self.pairs_frame = set()  # same-scope AND inside a function frame
        self.pairs_coupled = set()  # same-scope AND the dataflow edge exists
        self.outers = set()       # regions that exist at all in this file
        self.inners = set()
        self.nodes = 0

    def walk(self, node, stack, in_frame=False):
        """stack: list of (region_label, crossed_a_scope_boundary, sib_loads)."""
        self.nodes += 1
        kinds = inner_kinds(node)
        if kinds:
            stored = names(node, "store")
        for kind in kinds:
            self.inners.add(kind)
            for label, crossed, sibs in stack:
                self.pairs_any.add((label, kind))
                if not crossed:
                    self.pairs_same.add((label, kind))
                    if in_frame:
                        self.pairs_frame.add((label, kind))
                    if stored & sibs:
                        self.pairs_coupled.add((label, kind))
        for child, label, sibs in self.children(node):
            if label is None:
                self.walk(child, stack, in_frame)
                continue
            self.outers.add(label)
            below = stack
            if label in SCOPE_REGIONS:
                below = [(l, True, s) for l, _, s in below]
            self.walk(child, below + [(label, False, sibs)],
                      in_frame or label in FRAME_REGIONS)

    def children(self, node):
        """Yield (child, region_label_or_None, sibling_loads) per direct child."""
        cls = type(node).__name__
        empty = frozenset()

        if cls in COMP_KIND:
            kind = COMP_KIND[cls]
            # The element/key/value and every `if` run inside the implicit
            # function; so does every `iter` EXCEPT the first, which Python
            # evaluates in the enclosing scope before calling it. Labelling
            # that one as a scope-opening region would report a crossing that
            # the interpreter does not make.
            for field in ("elt", "key", "value"):
                sub = getattr(node, field, None)
                if sub is not None:
                    yield sub, f"{kind}.elt", empty
            for i, gen in enumerate(node.generators):
                yield gen.target, None, empty
                yield gen.iter, (None if i == 0 else f"{kind}.iter"), empty
                for cond in gen.ifs:
                    yield cond, f"{kind}.if", empty
            return

        if cls in ("FunctionDef", "AsyncFunctionDef"):
            prefix = "gen" if is_generator(node) else (
                "def" if cls == "FunctionDef" else "adef")
            if cls == "AsyncFunctionDef" and is_generator(node):
                prefix = "agen"
            for stmt in node.body:
                yield stmt, f"{prefix}.body", empty
            for field in ("args", "decorator_list", "returns", "type_params"):
                for sub in as_nodes(getattr(node, field, None)):
                    yield sub, None, empty
            return

        if cls == "ExceptHandler":
            for stmt in node.body:
                yield stmt, node._owner_region, node._owner_sibs
            for sub in as_nodes(node.type):
                yield sub, None, empty
            return

        for field, value in ast.iter_fields(node):
            if cls in ("Try", "TryStar") and field == "handlers":
                owner = "try.handler" if cls == "Try" else "try*.handler"
                sibs = frozenset(sibling_loads(node, "handlers"))
                for handler in value:
                    handler._owner_region = owner
                    handler._owner_sibs = sibs
                    yield handler, None, empty
                continue
            label = REGIONS.get((cls, field))
            sibs = (frozenset(sibling_loads(node, field)) if label else empty)
            for sub in as_nodes(value):
                yield sub, label, sibs


def as_nodes(value):
    if isinstance(value, ast.AST):
        return [value]
    if isinstance(value, list):
        return [v for v in value if isinstance(v, ast.AST)]
    return []


# ---------------------------------------------------------------- report

def scan_corpus(root):
    # Top level only, deliberately. `tests/probe` has a `tools/` subdirectory
    # holding the instruments themselves; recursing counts 19 measuring tools
    # as if they were corpus and inflates every denominator on this page.
    files = sorted(p for p in root.glob("*.py") if p.is_file())
    cell_files = Counter()      # (outer, inner) -> files, same-scope
    cell_files_any = Counter()
    cell_files_frame = Counter()
    cell_files_coupled = Counter()
    outer_files = Counter()
    inner_files = Counter()
    nodes = 0
    unparsed = []
    for path in files:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="strict"))
        except (SyntaxError, ValueError, UnicodeDecodeError) as exc:
            unparsed.append((path, f"{type(exc).__name__}: {exc}"))
            continue
        scan = Scan()
        scan.walk(tree, [])
        nodes += scan.nodes
        for cell in scan.pairs_same:
            cell_files[cell] += 1
        for cell in scan.pairs_any:
            cell_files_any[cell] += 1
        for cell in scan.pairs_frame:
            cell_files_frame[cell] += 1
        for cell in scan.pairs_coupled:
            cell_files_coupled[cell] += 1
        for label in scan.outers:
            outer_files[label] += 1
        for kind in scan.inners:
            inner_files[kind] += 1
    return dict(files=files, nodes=nodes, unparsed=unparsed,
                cells=cell_files, cells_any=cell_files_any,
                cells_frame=cell_files_frame, cells_coupled=cell_files_coupled,
                outer=outer_files, inner=inner_files)


# Cells that this tool's OWN axis split makes unreachable, as opposed to ones
# Python forbids. `def.body` is defined here as a function whose scope does not
# yield -- a yielding one is `gen.body` -- so `def.body > yield` can never be
# populated by any corpus and is not a gap in anything. Recording it here
# rather than letting it fall out as a GAP keeps the distinction between "no
# program does this" and "no program CAN do this" on the tool's own side.
AXIS_IMPOSSIBLE = {("def.body", "yield"), ("adef.body", "yield")}


def classify_zero(outer, inner, res):
    if (outer, inner) in AXIS_IMPOSSIBLE:
        return "VACUOUS-BY-AXIS"
    if res["outer"][outer] == 0:
        return "EMPTY-OUTER"
    if res["inner"][inner] == 0:
        return "EMPTY-INNER"
    return "GAP"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("roots", nargs="+", type=pathlib.Path)
    ap.add_argument("--gaps-only", action="store_true",
                    help="skip the matrix, print only the named empty cells")
    ap.add_argument("--allow-unparsed", action="append", default=[],
                    metavar="NAME",
                    help="a file that is EXPECTED not to parse, named one at a "
                         "time. Not a blanket switch: `tests/golden/errors` "
                         "contains parse_error.py, which asserts that lyc "
                         "rejects invalid syntax, so it is legitimately outside "
                         "this tool's domain -- but any OTHER rejection means "
                         "the host grammar is wrong and the denominator with "
                         "it, so it must still abort")
    args = ap.parse_args()

    print(f"nestgrid: host grammar = CPython "
          f"{sys.version_info.major}.{sys.version_info.minor}."
          f"{sys.version_info.micro}")
    print(f"axes: {len(OUTER_ORDER)} outer regions x {len(INNER_ORDER)} inner "
          f"kinds = {len(OUTER_ORDER) * len(INNER_ORDER)} cells per corpus")
    print("axis list is NOT exhaustive (no if/match/assert/decorator, no "
          "dataflow): every gap count below is a LOWER BOUND\n")

    refused = 0
    results = {}
    for root in args.roots:
        if not root.is_dir():
            print(f"REFUSED: {root} is not a directory", file=sys.stderr)
            return 2
        res = scan_corpus(root)
        results[root] = res
        # Denominators FIRST. An empty cell reads the same as an unscanned one.
        print(f"=== {root} ===")
        print(f"  files scanned : {len(res['files']) - len(res['unparsed'])}"
              f" of {len(res['files'])} *.py")
        print(f"  ast nodes     : {res['nodes']}")
        print(f"  unparsed      : {len(res['unparsed'])}")
        for path, why in res["unparsed"]:
            ok = path.name in args.allow_unparsed
            print(f"    {'-' if ok else '!'} {path}: {why}"
                  f"{'  (allowed)' if ok else ''}")
            refused += 0 if ok else 1
        print()

    if refused:
        print(f"REFUSED: {refused} file(s) did not parse under this host "
              f"grammar and were not named in --allow-unparsed. The corpus is "
              f"CPython 3.14 syntax; a rejection here means the denominator "
              f"above is wrong, not that those files have no findings. Re-run "
              f"with python3.14, or name each expected rejection explicitly.",
              file=sys.stderr)
        return 2

    for root, res in results.items():
        print(f"########## {root} ##########")
        if not args.gaps_only:
            print_matrix(res)
        print_gaps(res)
        print()
    return 0


def print_matrix(res):
    width = max(len(o) for o in OUTER_ORDER) + 1
    print("counts are FILES with the pair nested this way, in the same scope.")
    print("  ~  every one of them is at module level, outside a function frame")
    print("  !  NONE of them carries the dataflow edge (a name stored in the "
          "inner construct and read in a sibling region) -- the edge the "
          "shipped SIGSEGV needs")
    print(" " * width + "".join(f"{i[:5]:>7}" for i in INNER_ORDER))
    for outer in OUTER_ORDER:
        row = f"{outer:<{width}}"
        for inner in INNER_ORDER:
            cell = (outer, inner)
            n = res["cells"][cell]
            if not n:
                row += f"{'.':>7}"
                continue
            mark = "" if res["cells_frame"][cell] else "~"
            if not res["cells_coupled"][cell]:
                mark += "!"
            row += f"{str(n) + mark:>7}"
        print(row + f"   [{res['outer'][outer]} files have this region]")
    print()


def print_gaps(res):
    buckets = {"GAP": [], "GAP-COUPLED": [], "VACUOUS-BY-AXIS": [],
               "EMPTY-OUTER": [], "EMPTY-INNER": []}
    nonzero = 0
    for outer in OUTER_ORDER:
        for inner in INNER_ORDER:
            cell = (outer, inner)
            if res["cells"][cell]:
                if res["cells_coupled"][cell]:
                    nonzero += 1
                else:
                    buckets["GAP-COUPLED"].append(cell)
                continue
            buckets[classify_zero(outer, inner, res)].append(cell)
    total = len(OUTER_ORDER) * len(INNER_ORDER)
    print(f"cells: {total} total | {nonzero} covered WITH the dataflow edge | "
          f"{len(buckets['GAP-COUPLED'])} GAP-COUPLED | "
          f"{len(buckets['GAP'])} GAP | "
          f"{len(buckets['EMPTY-OUTER'])} vacuous (outer region absent) | "
          f"{len(buckets['EMPTY-INNER'])} vacuous (inner kind absent) | "
          f"{len(buckets['VACUOUS-BY-AXIS'])} vacuous (this tool's axis split)")
    print("GAP         = both features occur in this corpus, never nested "
          "this way.")
    print("GAP-COUPLED = nested this way, but no file couples them by a name "
          "stored inside and read in a sibling region. `try.body > for` is "
          "here, and that is where the shipped SIGSEGV lives.")
    print("Constructibility is NOT checked here: some cells cannot be written "
          "at all. nestwitness.py decides that by compiling a witness.")
    print("Ranked by min(files with outer, files with inner) -- a gap between "
          "two COMMON features is the one that hides a shipped defect.\n")
    for name in ("GAP-COUPLED", "GAP"):
        ranked = sorted(buckets[name], reverse=True,
                        key=lambda c: (min(res["outer"][c[0]],
                                           res["inner"][c[1]]),
                                       res["outer"][c[0]] + res["inner"][c[1]]))
        for outer, inner in ranked:
            cell = (outer, inner)
            bits = []
            if res["cells"][cell]:
                bits.append(f"{res['cells'][cell]} file(s) nest it, "
                            f"{res['cells_frame'][cell]} in a frame, "
                            f"0 coupled")
            elif res["cells_any"][cell]:
                bits.append(f"{res['cells_any'][cell]} file(s) only across a "
                            f"scope boundary")
            print(f"  {name:<11} {outer:<14} > {inner:<12} "
                  f"outer in {res['outer'][outer]:>3} files, "
                  f"inner in {res['inner'][inner]:>3} files"
                  + ("  (" + "; ".join(bits) + ")" if bits else ""))


if __name__ == "__main__":
    sys.exit(main())
