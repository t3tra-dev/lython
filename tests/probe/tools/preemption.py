#!/usr/bin/env python3
"""Static replay of findDeallocatorForValueGroup over every call-result list in
the runtime manifests, to count mechanism (B) -- PREEMPTION -- which the
in-compiler census (LYTHON_DEALLOC_CENSUS=1) structurally cannot see.

The census has counters for ambiguity (A), contract-aware resolution and the
two fallbacks.  It has none for (B): in the selection loop
(common/Ownership.cpp:486-501) a strictly longer `inputTypes` wins outright AND
RESETS `ambiguous`, so a preemption leaves no trace at all -- not even the tie it
overrode.  A counter cannot be added here because common/Ownership.cpp is owned
by another track, so this replays the algorithm against the same inputs instead.

Why NOT key on the leading value type: an arity-1 key cannot name an arity-3
tie; that error folded 63 container-tie exits into the width-2 bucket once
already (HandleWidthRegistry.h).  Groups are keyed on the whole matched
inputTypes list.

Why the call-result list is the right value range: `collectRuntimeResourceGroups`
is only ever fed `call.getResults()` (common/Ownership.cpp:1260), and it walks
offsets with `++offset` on a failed match, so every suffix of every call result
list is a probe.

INPUT VERIFICATION: every stage prints what it actually consumed and exits
nonzero if any stage consumed nothing.  A tool that answers where it should
refuse has been the failure mode five times in this investigation.
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


def split_types(text):
    text = text.strip()
    if not text or text == "()":
        return []
    if text.startswith("(") and balanced(text, 0) == len(text):
        text = text[1:-1]
    return re.findall(TYPE, text)


def param_types(text):
    """Types of a parameter list: the type after each top-level ':'."""
    return re.findall(r":\s*(" + TYPE + r")", text)


FUNC = re.compile(r"func\.func\s+(?:private\s+)?@([\w$.]+)\s*\(")
CALL = re.compile(r"func\.call\s+@([\w$.]+)\s*\(")

root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".")
mdir = root / "src/lython/runtime/modules"
paths = sorted(mdir.glob("*.mlir"))

print("== stage 1: input ==")
if not paths:
    sys.exit(f"REFUSING: no manifests under {mdir}")
total_bytes = 0
texts = {}
for p in paths:
    t = p.read_text()
    texts[p] = t
    total_bytes += len(t)
    print(f"   {p.name:22s} {len(t):9d} B  {t.count(chr(10)):6d} lines")
print(f"   files={len(paths)} bytes={total_bytes}")
if total_bytes == 0:
    sys.exit("REFUSING: manifests are empty")

# ---------------------------------------------------------------- stage 2
# Signatures: name -> (params, results, attrs).
sigs = {}
unparsed_funcs = []
for p, t in texts.items():
    for line in t.split("\n"):
        m = FUNC.search(line)
        if not m:
            continue
        open_paren = m.end() - 1
        close = balanced(line, open_paren)
        if close is None:
            unparsed_funcs.append((p.name, line[:90]))
            continue
        params = line[open_paren:close]
        rest = line[close:]
        arrow = rest.find("->")
        attr = rest.find("attributes")
        if arrow == -1 or (attr != -1 and arrow > attr):
            results = []
        else:
            tail = rest[arrow + 2:]
            stop = tail.find("attributes")
            if stop != -1:
                tail = tail[:stop]
            tail = tail.strip().rstrip("{").strip()
            results = split_types(tail)
        sigs[m.group(1)] = (param_types(params), results, rest)

print("== stage 2: signatures ==")
print(f"   func.func parsed={len(sigs)} unparsed={len(unparsed_funcs)}")
for n, l in unparsed_funcs[:5]:
    print(f"   UNPARSED {n}: {l}")
if not sigs:
    sys.exit("REFUSING: parsed zero signatures")

# ---------------------------------------------------------------- stage 3
# Deallocator table + canonical shapes, mirroring buildRuntimeDeallocators.
deallocs = []  # (contract, inputTypes)
shapes = {}    # contract -> shapeTypes
for name, (params, results, rest) in sigs.items():
    c = re.search(r'ly\.runtime\.contract = "([^"]+)"', rest)
    contract = c.group(1) if c else None
    if "ly.runtime.deallocator" in rest and contract:
        deallocs.append((contract, name, params))
    if "ly.runtime.shape" in rest and contract:
        shapes[contract] = results

table = []
for contract, name, inputs in deallocs:
    shape = shapes.get(contract) or inputs
    table.append({"contract": contract, "fn": name,
                  "inputs": inputs, "shape": shape})

print("== stage 3: deallocator table ==")
print(f"   deallocators={len(table)} shapes={len(shapes)}")
by_iface = collections.defaultdict(set)
for d in table:
    if len(d["inputs"]) == 1:
        by_iface[d["inputs"][0]].add(d["contract"])
ties = {w: cs for w, cs in by_iface.items() if len(cs) > 1}
print(f"   single-input ifaces={len(by_iface)} tied={len(ties)}")
if not table:
    sys.exit("REFUSING: deallocator table is empty")


def matches(values, offset, types):
    if offset + len(types) > len(values):
        return False
    return all(values[offset + i] == t for i, t in enumerate(types))


def select(values, offset):
    """findDeallocatorForValueGroup (contract-less). Returns
    (matched, ambiguous, all_matching)."""
    matched, matched_shape, ambiguous = None, 0, False
    hits = []
    for d in table:
        if not matches(values, offset, d["inputs"]):
            continue
        hits.append(d)
        shape = (len(d["shape"])
                 if len(d["shape"]) > len(d["inputs"])
                 and matches(values, offset, d["shape"]) else 0)
        if (matched is None
                or len(d["inputs"]) > len(matched["inputs"])
                or (len(d["inputs"]) == len(matched["inputs"])
                    and shape > matched_shape)):
            matched, matched_shape, ambiguous = d, shape, False
            continue
        if (len(d["inputs"]) == len(matched["inputs"])
                and shape == matched_shape):
            ambiguous = True
    return (None if ambiguous else matched), ambiguous, hits


# ---------------------------------------------------------------- stage 4
# Every call-result list in the manifests is a scanned value range.
ranges = collections.Counter()
callees_of = collections.defaultdict(collections.Counter)
call_sites = 0
unknown_callee = collections.Counter()
for p, t in texts.items():
    for line in t.split("\n"):
        for m in CALL.finditer(line):
            call_sites += 1
            callee = m.group(1)
            if callee not in sigs:
                unknown_callee[callee] += 1
                continue
            res = tuple(sigs[callee][1])
            if res:
                ranges[res] += 1
                callees_of[res][callee] += 1

print("== stage 4: value ranges ==")
print(f"   func.call sites={call_sites} "
      f"distinct result lists={len(ranges)} "
      f"with-results calls={sum(ranges.values())} "
      f"unknown callees={len(unknown_callee)}")
if not ranges:
    sys.exit("REFUSING: found zero call result lists")

# ---------------------------------------------------------------- stage 5
resolved = collections.Counter()
ambiguous = collections.Counter()
preempted = collections.Counter()   # (loser iface, winner contract) -> n
preempt_detail = collections.defaultdict(collections.Counter)
preempt_sites = collections.defaultdict(collections.Counter)
for values, n in ranges.items():
    values = list(values)
    offset = 0
    while offset < len(values):
        matched, amb, hits = select(values, offset)
        key = None
        if amb:
            # The census keys ambiguity on the matched interface.
            _, _, h = select(values, offset)
            iface = None
            for d in table:
                if matches(values, offset, d["inputs"]):
                    iface = tuple(d["inputs"])
                    break
            ambiguous[iface] += n
            offset += 1
            continue
        if matched is None:
            offset += 1
            continue
        # (B): a strictly longer interface won where shorter ones also matched.
        losers = {tuple(d["inputs"]) for d in hits
                  if len(d["inputs"]) < len(matched["inputs"])}
        loser_contracts = {d["contract"] for d in hits
                           if len(d["inputs"]) < len(matched["inputs"])}
        if losers:
            for l in losers:
                preempted[(l, matched["contract"])] += n
                preempt_detail[l][matched["contract"]] += n
                preempt_sites[(l, matched["contract"])][
                    (tuple(values), offset)] += n
        resolved[matched["contract"]] += n
        offset += len(matched["inputs"])

print()
print("== (A) AMBIGUOUS exits, by tied interface (per manifest call, static) ==")
if not ambiguous:
    print("   none")
for k, v in sorted(ambiguous.items(), key=lambda kv: -kv[1]):
    cs = sorted(by_iface.get(k[0], set())) if k and len(k) == 1 else []
    print(f"   {str(k):58s} = {v:6d}   {','.join(cs)}")

print()
print("== (B) PREEMPTION: shorter interface matched, longer one took it ==")
if not preempted:
    print("   none")
for loser, winners in sorted(preempt_detail.items(),
                             key=lambda kv: -sum(kv[1].values())):
    owners = sorted(by_iface.get(loser[0], set())) if len(loser) == 1 else []
    print(f"   loser {str(loser):40s} total={sum(winners.values()):6d}")
    if owners:
        print(f"        one-lane owners of that width: {', '.join(owners)}")
    for w, n in sorted(winners.items(), key=lambda kv: -kv[1]):
        print(f"        preempted by {w:44s} {n:6d}")

print()
print("== (B) ADJUDICATION ==")
print("A preemption is only a DEFECT if the winner is not the value's true")
print("owner.  For a genuine exception triple the leading memref<3xi64> IS the")
print("exception header, so BaseException winning is CORRECT and required.")
print("Each distinct (result list, offset) is printed with its callees so the")
print("kind can be decided from the DECLARATION, never from the width.")
for (loser, winner), sites in sorted(
        preempt_sites.items(), key=lambda kv: -sum(kv[1].values())):
    print(f"\n  loser {loser} preempted by {winner}:")
    for (values, offset), n in sorted(sites.items(), key=lambda kv: -kv[1]):
        print(f"    n={n:5d} offset={offset} results={list(values)}")
        ex = callees_of[values]
        shown = 0
        for callee, cn in sorted(ex.items(), key=lambda kv: -kv[1]):
            rest = sigs[callee][2]
            c = re.search(r'ly\.runtime\.contract = "([^"]+)"', rest)
            owned = re.search(r'ly\.ownership\.owned_results = \[([^\]]*)\]',
                              rest)
            rc = re.search(
                r'ly\.ownership\.owned_result_contracts = \[([^\]]*)\]', rest)
            print(f"          {callee:44s} x{cn:4d} "
                  f"contract={c.group(1) if c else '-'} "
                  f"owned=[{owned.group(1) if owned else '-'}] "
                  f"result_contracts=[{rc.group(1) if rc else '-'}]")
            shown += 1
            if shown >= 12:
                print(f"          ... {len(ex) - shown} more callees")
                break

# ---------------------------------------------------------------- stage 6
# DOMAIN CORRECTION.  Stage 4 only saw calls written INSIDE the manifests, which
# misses every manifest function that is called from emitted user code -- e.g.
# `LyReadyIntAwaitable_Await`, whose result list ties at width 3 and has no
# intra-manifest caller at all.  The value range a user-code call presents is
# the callee's declared result list, so every function with results is a range.
#
# Why NOT report a raw preemption count here: for a genuine exception triple the
# leading memref<3xi64> IS the exception header, so "a shorter interface also
# matched" is true of every correct case.  The decidable question is whether the
# winning contract is one the CALLEE declares -- its own `ly.runtime.contract`
# or its `ly.runtime.result_contract`.  If it is neither, the width has selected
# a foreign deallocator and that is mechanism (B) for real.
print()
print("== stage 6: (B) over ALL function result lists, adjudicated by "
      "declaration ==")
checked = 0
benign = collections.Counter()
inherited = collections.Counter()
suspicious = []
contracts_with_dealloc = {d["contract"] for d in table}
for name, (params, results, rest) in sorted(sigs.items()):
    if not results:
        continue
    checked += 1
    own = re.search(r'ly\.runtime\.contract = "([^"]+)"', rest)
    own = own.group(1) if own else None
    rc = re.search(r'ly\.runtime\.result_contract = "([^"]+)"', rest)
    rc = rc.group(1) if rc else None
    rcs = re.findall(r'"([^"]+)"',
                     (re.search(
                         r'ly\.ownership\.owned_result_contracts = \[([^\]]*)\]',
                         rest) or re.match("", "")).group(1)
                     if re.search(
                         r'ly\.ownership\.owned_result_contracts = \[([^\]]*)\]',
                         rest) else "")
    declared = {c for c in [own, rc, *rcs] if c}
    offset = 0
    while offset < len(results):
        matched, amb, hits = select(results, offset)
        if amb or matched is None:
            offset += 1
            continue
        shorter = [d for d in hits
                   if len(d["inputs"]) < len(matched["inputs"])]
        if shorter:
            # Benign in two ways, and both have to be allowed for or the
            # exception family alone produces 209 false positives:
            #   1. the winner is a contract the callee declares; or
            #   2. every contract the callee declares has NO deallocator of its
            #      own, so it legitimately inherits one structurally -- which is
            #      how all ~70 exception subclasses reach LyBaseException_DecRef.
            # Only a callee whose OWN contract declares a deallocator, losing to
            # a different one, is mechanism (B).
            own_dealloc = declared & contracts_with_dealloc
            if matched["contract"] in declared:
                benign[matched["contract"]] += 1
            elif declared and not own_dealloc:
                inherited[matched["contract"]] += 1
            else:
                suspicious.append(
                    (name, offset, matched["contract"], declared,
                     [d["contract"] for d in shorter], list(results)))
        offset += len(matched["inputs"])

print(f"   functions with results checked = {checked}")
print(f"   preemptions where the winner IS declared by the callee (benign):")
for c, n in sorted(benign.items(), key=lambda kv: -kv[1]):
    print(f"        {c:44s} {n:5d}")
if not benign:
    print("        none")
print(f"   preemptions where the callee's contract declares NO deallocator, "
      f"so the winner is inherited by design:")
for c, n in sorted(inherited.items(), key=lambda kv: -kv[1]):
    print(f"        {c:44s} {n:5d}")
if not inherited:
    print("        none")
print(f"   contracts declaring a deallocator = {len(contracts_with_dealloc)}")
print(f"   SUSPICIOUS (callee owns a deallocator, a different one won) "
      f"= {len(suspicious)}")
for name, off, win, decl, losers, res in suspicious:
    print(f"        {name} offset={off}")
    print(f"            winner={win} declared={sorted(decl) or '-'} "
          f"losers={sorted(set(losers))}")
    print(f"            results={res}")

# ---------------------------------------------------------------- stage 7
# (A) per function, per tied width: which callee's result loses its group.
# The census gives counts only; this names them, so "18 ambiguous exits at
# width 3" can be checked against a list of declarations rather than believed.
print()
print("== stage 7: (A) AMBIGUOUS by tied width, naming the callee ==")
amb_by_width = collections.defaultdict(list)
for name, (params, results, rest) in sorted(sigs.items()):
    if not results:
        continue
    own = re.search(r'ly\.runtime\.contract = "([^"]+)"', rest)
    own = own.group(1) if own else "-"
    rc = re.search(r'ly\.runtime\.result_contract = "([^"]+)"', rest)
    rc = rc.group(1) if rc else "-"
    owned = re.search(r'ly\.ownership\.owned_results = \[([^\]]*)\]', rest)
    offset = 0
    while offset < len(results):
        matched, amb, hits = select(results, offset)
        if amb:
            iface = next((tuple(d["inputs"]) for d in table
                          if matches(results, offset, d["inputs"])), None)
            if iface and len(iface) == 1:
                amb_by_width[iface[0]].append(
                    (name, offset, own, rc,
                     owned.group(1) if owned else "-"))
            offset += 1
            continue
        offset += len(matched["inputs"]) if matched else 1

for w in sorted(amb_by_width, key=lambda k: -len(amb_by_width[k])):
    rows = amb_by_width[w]
    owners = sorted(by_iface.get(w, set()))
    print(f"\n  {w}  sites={len(rows)}  owners={', '.join(owners)}")
    # Only the ones that declare an OWNED result can leak: a borrowed result
    # carries no release obligation, so an ambiguous exit there costs nothing.
    owning = [r for r in rows if r[4] != "-"]
    print(f"       of which declare ly.ownership.owned_results: "
          f"{len(owning)}  <-- only these can leak")
    for name, off, own, rcn, ow in owning[:24]:
        print(f"       {name:42s} off={off} contract={own} "
              f"result_contract={rcn} owned=[{ow}]")
    if len(owning) > 24:
        print(f"       ... {len(owning) - 24} more")

print()
print("== resolved, by contract (denominator per width) ==")
for c, v in sorted(resolved.items(), key=lambda kv: -kv[1]):
    iface = next((tuple(d["inputs"]) for d in table if d["contract"] == c), ())
    print(f"   {c:34s} {v:6d}   iface={''.join(str(iface))}")
