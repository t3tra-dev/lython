#!/usr/bin/env python3
"""Per-program traffic through each tied release interface, from the compiler's
own LYTHON_DEALLOC_CENSUS.

`preemption.py` measures the STATIC surface (which manifest call-result lists
could tie).  This measures TRAFFIC (which ties a real compile actually reaches).
The project's rule 8 is that the algebra does not tell you whether a path
carries traffic, so both are needed and neither substitutes.

INPUT VERIFICATION, because five instrument defects this week were all "answers
where it should refuse":
  * the census prints from a STATIC DESTRUCTOR, so a killed or crashed compile
    emits NOTHING -- zero census lines is indistinguishable from a clean compile
    unless the exit code is checked.  Every row therefore carries rc, and a row
    with no `[DEALLOC]` summary line at all is printed as MISSING, never as 0.
  * the binary is hashed once and reported, because an A/B where both arms were
    the same rebuilt binary has already produced a false "no difference".
  * `python3 -u` at the top of every invocation; block-buffered output loses the
    whole buffer when killed and 0 bytes then reads as a finding.

Usage:  python3 -u tests/probe/tools/tietraffic.py <lyc> <prog.py> [prog.py ...]
"""
import collections
import hashlib
import os
import pathlib
import re
import subprocess
import sys

# Ties as of this tree; read from the manifests by tiecensus.py, repeated here
# only for labelling.  A width absent from this map still gets reported.
OWNERS = {
    "(memref<2xi64>)": "int / str / nullcontext",
    "(memref<3xi64>)": "bool / float / ReadyIntAwaitable",
    "(memref<4xi64>)": "Counter / AsyncCounter / ReadyAsyncCounter",
    "(memref<5xi64>)": "range / range_iterator / CoroutineType",
    "(memref<8xi64>)": "dict / function / _io.{4} / EventLoop",
}

if len(sys.argv) < 3:
    sys.exit(__doc__)
lyc = pathlib.Path(sys.argv[1]).resolve()
progs = [pathlib.Path(p) for p in sys.argv[2:]]
if not lyc.exists():
    sys.exit(f"REFUSING: no such binary {lyc}")

digest = hashlib.sha256(lyc.read_bytes()).hexdigest()
print(f"binary   {lyc}")
print(f"sha256   {digest}")
print(f"load     {os.getloadavg()}")
missing = [p for p in progs if not p.exists()]
if missing:
    sys.exit(f"REFUSING: missing inputs {missing}")
print(f"programs {len(progs)}")
print()

SUMMARY = re.compile(
    r"\[DEALLOC\] empty_name=(\d+) fallback_450=(\d+) "
    r"contract_aware_ambiguous=(\d+) declared_name_absent=(\d+)")
AMB = re.compile(r"\[DEALLOC\] ambiguous (\S+) (\([^=]*\)) = (\d+)")
RES = re.compile(r"\[DEALLOC\] resolved (\S+) = (\d+)")

rows = []
for prog in progs:
    env = dict(os.environ, LYTHON_DEALLOC_CENSUS="1")
    out = subprocess.run(
        [str(lyc), str(prog), "-o", os.devnull],
        capture_output=True, text=True, env=env)
    err = out.stderr
    summary = SUMMARY.search(err)
    amb = collections.Counter()
    for m in AMB.finditer(err):
        amb[(m.group(1), m.group(2))] += int(m.group(3))
    res = {m.group(1): int(m.group(2)) for m in RES.finditer(err)}
    rows.append({
        "prog": prog.name, "rc": out.returncode,
        "summary": summary.groups() if summary else None,
        "amb": amb, "res": res,
        "census_lines": err.count("[DEALLOC]"),
    })

# ------------------------------------------------------------------ report
print("== per-program census presence (a compile that died prints NOTHING) ==")
for r in rows:
    state = "ok" if r["summary"] else "MISSING-CENSUS"
    print(f"   {r['prog']:44s} rc={r['rc']:3d} "
          f"[DEALLOC] lines={r['census_lines']:4d}  {state}")
usable = [r for r in rows if r["summary"]]
print(f"   usable rows: {len(usable)} of {len(rows)}")
if not usable:
    sys.exit("REFUSING: no program produced a census; nothing to report")

print()
print("== (A) AMBIGUOUS exits per compile, by origin and tied interface ==")
keys = sorted({k for r in usable for k in r["amb"]})
if not keys:
    print("   none in any program")
for origin, iface in keys:
    vals = [r["amb"].get((origin, iface), 0) for r in usable]
    lo, hi = min(vals), max(vals)
    tag = "INVARIANT" if lo == hi else f"VARIES {lo}..{hi}"
    print(f"   {iface:22s} {origin:38s} {tag}")
    print(f"        owners: {OWNERS.get(iface, '(untied / multi-input)')}")
    for r, v in zip(usable, vals):
        mark = " <-- moves" if lo != hi and v != lo else ""
        print(f"        {r['prog']:44s} {v:6d}{mark}")

print()
print("== resolved per compile, by contract (the DENOMINATOR per width) ==")
contracts = sorted({c for r in usable for c in r["res"]})
width = {}
print(f"   {'contract':34s} " + " ".join(f"{r['prog'][:14]:>15s}"
                                         for r in usable))
for c in contracts:
    vals = [r["res"].get(c, 0) for r in usable]
    print(f"   {c:34s} " + " ".join(f"{v:15d}" for v in vals))

print()
print("== summary counters ==")
print(f"   {'program':44s} {'empty':>7s} {'fb450':>7s} "
      f"{'ca_amb':>7s} {'name_absent':>12s}")
for r in usable:
    e, f, ca, na = r["summary"]
    print(f"   {r['prog']:44s} {e:>7s} {f:>7s} {ca:>7s} {na:>12s}")
print()
print(f"load at end {os.getloadavg()}")
