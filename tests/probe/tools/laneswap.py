#!/usr/bin/env python3
"""Replay `collectRuntimeResourceGroups` over the manifests, and answer what a
one-lane conversion would do to ambiguity BEFORE doing it.

This is the instrument that adjudicated `builtins.int` (HandleWidthRegistry.h,
the deferral block): it re-runs the selection loop with a contract's release
interface and canonical shape rewritten to a single lane, and reports the
difference in ambiguous exits.  The int track wrote it, quoted it in a commit
message and did not keep it, so the next contract had to write it again -- which
is the reason it is a file this time.

Why the offset advance matters, and why this is NOT preemption.py's loop:
`collectRuntimeResourceGroups` advances by `inputTypes + views`
(common/Ownership.cpp:1015-1017), not by `inputTypes`.  A multi-lane contract
whose shape tail matches therefore CONSUMES its interior lanes, so those offsets
are never probed at all.  One-laning a contract shortens the span to 1 and
exposes offsets that were previously hidden -- which is a second-order effect of
the conversion that a fixed `offset += len(inputs)` replay cannot see.
preemption.py measures mechanism (B) and advances the other way on purpose; the
two tools do not substitute for each other.

Why report shape-DECIDED resolutions separately: a resolution that survives the
conversion is not the same as one that never depended on the shape score.  The
count answers "how much protection is actually being spent", which for int was
110 of 110 and is the figure the width choice has to cover.

Usage:
    python3 -u tests/probe/tools/laneswap.py <repo-root> [ARM ...]
    ARM := <contract>@<width>   e.g. builtins.str@17
                                     builtins.int@2   (one-lane, width kept)
Multiple ARMs in one invocation are applied together (that is how a
"move all three off width 2" question is asked).  With no ARM it prints the
unmodified tree, which is the baseline every arm is read against.

CALIBRATION -- run these three before trusting a new answer, because they are
the only published figures this can be checked against (HandleWidthRegistry.h,
int's deferral block):

    (no arm)          total ambiguous 80,  on (memref<2xi64>) 4,
                      resolved builtins.int 110, shape-decided 110
    builtins.int@15   total 80,  width-2 4,  110 collapsed
    builtins.int@2    total 190, width-2 114

If those three do not come out, the replay has drifted from
`findDeallocatorForValueGroup` and nothing else it prints is worth reading.
"""
import collections
import pathlib
import re
import sys


def balanced(text, start):
    """Index just past the ')' matching the '(' at `start`, or None."""
    if start >= len(text) or text[start] != "(":
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return i + 1
    return None


TYPE = r"memref<[^<>]*>|complex<[^<>]*>|i\d+|f\d+|index|none|!\w[\w.]*"
FUNC = re.compile(r"func\.func\s+(?:private\s+)?@([\w$.]+)\s*\(")


def split_types(text):
    text = text.strip()
    if not text or text == "()":
        return []
    if text.startswith("(") and balanced(text, 0) == len(text):
        text = text[1:-1]
    return re.findall(TYPE, text)


def param_types(text):
    return re.findall(r":\s*(" + TYPE + r")", text)


if len(sys.argv) < 2:
    sys.exit(__doc__)
root = pathlib.Path(sys.argv[1])
arms = []
for spec in sys.argv[2:]:
    if "@" not in spec:
        sys.exit(f"REFUSING: bad arm {spec!r}, want <contract>@<width>")
    contract, width = spec.rsplit("@", 1)
    arms.append((contract, int(width)))

mdir = root / "src/lython/runtime/modules"
paths = sorted(mdir.glob("*.mlir"))
if not paths:
    sys.exit(f"REFUSING: no manifests under {mdir}")

# ------------------------------------------------------------------ stage 1
texts = {p: p.read_text() for p in paths}
total_bytes = sum(len(t) for t in texts.values())
print(f"== stage 1: input ==  files={len(paths)} bytes={total_bytes}")
if total_bytes == 0:
    sys.exit("REFUSING: manifests are empty")

# ------------------------------------------------------------------ stage 2
# name -> (params, results, tail-of-line).  Later files do not overwrite an
# earlier definition, matching a module walk that sees each symbol once.
sigs = {}
order = []
unparsed = []
decl_lines = []
for p in paths:
    for line in texts[p].split("\n"):
        m = FUNC.search(line)
        if not m:
            continue
        close = balanced(line, m.end() - 1)
        if close is None:
            unparsed.append((p.name, line[:90]))
            continue
        rest = line[close:]
        arrow, attr = rest.find("->"), rest.find("attributes")
        if arrow == -1 or (attr != -1 and arrow > attr):
            results = []
        else:
            tail = rest[arrow + 2:]
            stop = tail.find("attributes")
            if stop != -1:
                tail = tail[:stop]
            results = split_types(tail.strip().rstrip("{").strip())
        name = m.group(1)
        decl_lines.append((name, results))
        if name not in sigs:
            order.append(name)
        sigs[name] = (param_types(line[m.end() - 1:close]), results, rest)
print(f"== stage 2: signatures ==  parsed={len(sigs)} unparsed={len(unparsed)}")
for n, l in unparsed[:5]:
    print(f"   UNPARSED {n}: {l}")
if not sigs:
    sys.exit("REFUSING: parsed zero signatures")

# ------------------------------------------------------------------ stage 3
# collectRuntimeDeallocators: every DECLARATION carrying the attribute, then
# canonical shapes joined by contract name, defaulting to the interface.
# Why dedup by SYMBOL here but not for the value ranges below: the compiler
# walks one linked module, where `LyBytes_DecRef` -- declared in four manifests
# -- is one func.func.  Keeping all four makes bytes tie with ITSELF and
# manufactures 17 ambiguous exits that no compile can have.  (The int block
# reports "37 deallocators", which is tiecensus's declaration count; the table
# that produced its 80/4/110 is 34.  The three figures are unaffected.)
decls = []
shapes = {}
seen_dealloc = set()
for p in paths:
    for line in texts[p].split("\n"):
        m = FUNC.search(line)
        if not m:
            continue
        close = balanced(line, m.end() - 1)
        if close is None:
            continue
        rest = line[close:]
        c = re.search(r'ly\.runtime\.contract = "([^"]+)"', rest)
        if not c:
            continue
        if "ly.runtime.deallocator" in rest and m.group(1) not in seen_dealloc:
            seen_dealloc.add(m.group(1))
            decls.append((c.group(1), m.group(1),
                          param_types(line[m.end() - 1:close])))
        if "ly.runtime.shape" in rest:
            shapes[c.group(1)] = sigs[m.group(1)][1]

table = [{"contract": c, "fn": f, "inputs": list(i),
          "shape": list(shapes.get(c) or i)} for c, f, i in decls]
baseline_table = [dict(d, inputs=list(d["inputs"]), shape=list(d["shape"]))
                  for d in table]
print(f"== stage 3: deallocators ==  {len(table)} decls, {len(shapes)} shapes")
if not table:
    sys.exit("REFUSING: deallocator table is empty")

def matches(values, offset, types):
    if offset + len(types) > len(values):
        return False
    return all(values[offset + i] == t for i, t in enumerate(types))


def select(values, offset, use_shape=True, table=None):
    """findDeallocatorForValueGroup, contract-less overload
    (common/Ownership.cpp:470).  Returns (matched_or_None, ambiguous, iface)."""
    table = globals()["table"] if table is None else table
    matched, matched_shape, ambiguous, iface = None, 0, False, None
    for d in table:
        if not matches(values, offset, d["inputs"]):
            continue
        if iface is None:
            iface = tuple(d["inputs"])
        shape = 0
        if (use_shape and len(d["shape"]) > len(d["inputs"])
                and matches(values, offset, d["shape"])):
            shape = len(d["shape"])
        if (matched is None
                or len(d["inputs"]) > len(matched["inputs"])
                or (len(d["inputs"]) == len(matched["inputs"])
                    and shape > matched_shape)):
            matched, matched_shape, ambiguous = d, shape, False
            continue
        if (len(d["inputs"]) == len(matched["inputs"])
                and shape == matched_shape):
            ambiguous = True
    if ambiguous:
        # The census keys the exit on the winner's interface, not the leading
        # value type: an arity-1 key cannot name an arity-3 tie.
        return None, True, tuple(matched["inputs"]) if matched else None
    return matched, False, iface


# ------------------------------------------------------------------ stage 4
# Value ranges: every function's own result list.  `collectRuntimeResourceGroups`
# is fed `call.getResults()`, whose types ARE the callee's result types, so the
# per-function list is the static surface.
#
# Why declaration LINES and not distinct symbols: a symbol re-declared in four
# manifests is four import sites, and this weights the surface the way the int
# measurement did (1180 lines, 1103 symbols).  It is a weighting choice, not a
# fact about the tree -- every conclusion below is a DIFFERENCE between arms, and
# the difference is the same under either weighting.  Stated because a reader
# comparing 1180 against a symbol count will otherwise think one is wrong.
ranges = [(n, tuple(r)) for n, r in decl_lines if r]
print(f"== stage 4: value ranges ==  declaration lines with results={len(ranges)}"
      f"  (distinct symbols={sum(1 for n in order if sigs[n][1])})")
if not ranges:
    sys.exit("REFUSING: found zero functions with results")

# ------------------------------------------------------------------ stage 5
# Applying an arm is TWO edits, not one, and doing only the first is what makes
# the model incoherent: a conversion flips the contract's release interface AND
# collapses its lanes wherever a signature spells them.  A tree with a one-lane
# `LyLong_DecRef` and 155 signatures still returning the triple does not exist.
#
# Why the spans are found by scanning rather than by substring-replacing the
# shape: the exception triple's tail IS `(memref<2xi64>, memref<?xi8>)`, i.e.
# `str`'s whole canonical shape, so a blind replace would collapse two thirds of
# every exception into a str lane and invent ambiguity that no conversion
# creates.  The baseline scan resolves that triple to BaseException at offset 0
# and consumes all three, so the tail is never a candidate -- which is exactly
# the compiler's own answer to the same question.
def collapse(values, contract, lane):
    out, offset = [], 0
    values = list(values)
    n = 0
    while offset < len(values):
        matched, amb, _ = select(values, offset, table=baseline_table)
        if matched is None:
            out.append(values[offset])
            offset += 1
            continue
        span = len(matched["inputs"])
        tail = matched["shape"][span:]
        if tail and matches(values, offset + span, tail):
            span += len(tail)
        if matched["contract"] == contract:
            out.append(lane)
            n += 1
        else:
            out.extend(values[offset:offset + span])
        offset += span
    return out, n


for contract, width in arms:
    lane = f"memref<{width}xi64>"
    hit = sum(1 for d in table if d["contract"] == contract)
    if not hit:
        sys.exit(f"REFUSING: arm {contract}@{width} matched no deallocator")
    for d in table:
        if d["contract"] == contract:
            d["inputs"], d["shape"] = [lane], [lane]
    collapsed = 0
    new_ranges = []
    for fn, values in ranges:
        v, n = collapse(values, contract, lane)
        collapsed += n
        new_ranges.append((fn, tuple(v)))
    ranges = new_ranges
    print(f"   ARM {contract} -> one lane {lane}: {hit} deallocator "
          f"declaration(s), {collapsed} lane group(s) collapsed in signatures")
    if collapsed == 0:
        sys.exit(f"REFUSING: arm {contract}@{width} collapsed nothing")

by_iface = collections.defaultdict(set)
for d in table:
    if len(d["inputs"]) == 1:
        by_iface[d["inputs"][0]].add(d["contract"])

ambiguous = collections.Counter()
resolved = collections.Counter()
shape_decided = collections.Counter()
for fn, values in ranges:
    values = list(values)
    offset = 0
    while offset < len(values):
        matched, amb, iface = select(values, offset)
        if amb:
            ambiguous[iface] += 1
            offset += 1
            continue
        if matched is None:
            offset += 1
            continue
        resolved[matched["contract"]] += 1
        # Would this resolution survive without the shape score?  Anything but
        # the same contract means the score decided it.
        plain, plain_amb, _ = select(values, offset, use_shape=False)
        if plain_amb or plain is None or plain["contract"] != matched["contract"]:
            shape_decided[matched["contract"]] += 1
        span = len(matched["inputs"])
        tail = matched["shape"][span:]
        if tail and matches(values, offset + span, tail):
            span += len(tail)
        offset += span

print()
print(f"== ambiguous exits ==  total={sum(ambiguous.values())}")
for k, v in sorted(ambiguous.items(), key=lambda kv: -kv[1]):
    owners = sorted(by_iface.get(k[0], ())) if k and len(k) == 1 else []
    print(f"   {str(k):58s} = {v:5d}   {','.join(owners)}")
print()
print(f"== resolved ==  total={sum(resolved.values())}   "
      f"shape-decided={sum(shape_decided.values())}")
for c, v in sorted(resolved.items(), key=lambda kv: -kv[1]):
    print(f"   {c:34s} {v:5d}   shape-decided {shape_decided[c]:5d}")
