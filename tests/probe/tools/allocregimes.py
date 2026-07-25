#!/usr/bin/env python3
"""Run probes under several allocator regimes, not just one.

`flaky.py` answers "is the outcome stable under repetition". This answers a
different question that repetition cannot: "is the outcome stable under a
DIFFERENT heap layout". They are not the same, and the distinction decides
whether a use-after-free family that stopped failing is actually fixed or merely
no longer landing on a reused block.

The regimes probe different things, so a shape passing all of them is a much
stronger statement than a shape passing one many times:

  plain             the system allocator, which reuses freed blocks eagerly --
                    a use-after-free here usually reads back plausible data
  guard-after       libgmalloc, guard page AFTER the block: catches overruns and
                    unmaps the block on free, so any later access faults
  guard-before      libgmalloc, guard page BEFORE the block: shifts every
                    allocation's alignment and catches underruns instead, so it
                    is a different LAYOUT as well as a different check
  scribble          system allocator filling freed memory with 0x55: a
                    use-after-free READ returns garbage rather than the old
                    value, which catches the reads that survive unmapping
                    because they never happen after the free in wall-clock order
  prescribble       system allocator filling fresh memory with 0xAA: catches a
                    read of a slot that was never initialised, which is how a
                    dropped store presents

A shape that passes plain but fails any guarded or scribbled regime is a live
memory-safety defect whose visibility depends on allocator behaviour -- the class
this corpus exists to make deterministic.

    python3 tests/probe/tools/allocregimes.py ./build/bin/lyc tests/probe/alias_*.py

Exit code is the number of probes failing in at least one regime.
"""

import argparse
import os
import pathlib
import re
import subprocess
import sys

CPY = "/opt/homebrew/Frameworks/Python.framework/Versions/3.14/bin/python3.14"

REGIMES = {
    "plain": {},
    "guard-after": dict(DYLD_INSERT_LIBRARIES="/usr/lib/libgmalloc.dylib",
                        MALLOC_FILL_SPACE="1"),
    "guard-before": dict(DYLD_INSERT_LIBRARIES="/usr/lib/libgmalloc.dylib",
                         MALLOC_PROTECT_BEFORE="1", MALLOC_FILL_SPACE="1"),
    "scribble": dict(MallocScribble="1"),
    "prescribble": dict(MallocPreScribble="1", MallocScribble="1"),
}

REFCOUNT = re.compile(r"Ly_(?:Inc|Dec)Ref observed non-positive refcount")


def run(lyc, case, env_extra, timeout=900.0):
    env = dict(os.environ)
    env.update(env_extra)
    # libgmalloc announces itself on stderr; drop that so it is not mistaken
    # for a diagnostic.
    try:
        r = subprocess.run([str(lyc), "jit", str(case)], capture_output=True,
                           text=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        return "timeout"
    err = "\n".join(l for l in r.stderr.splitlines()
                    if not l.startswith("GuardMalloc["))
    want = run.want
    if r.returncode == 0:
        return "ok" if r.stdout == want else f"SILENT({r.stdout.strip()})"
    if REFCOUNT.search(r.stdout + err):
        return "abort"
    if r.returncode < 0:
        return f"sig{-r.returncode}"
    return "reject"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("lyc", type=pathlib.Path)
    ap.add_argument("probes", nargs="+", type=pathlib.Path)
    ap.add_argument("-n", "--runs", type=int, default=3,
                    help="runs per regime (default 3)")
    args = ap.parse_args()
    lyc = args.lyc.resolve()

    names = list(REGIMES)
    width = max(len(p.name) for p in args.probes)
    print(f"{'probe':{width}}  " + "  ".join(f"{n:>12}" for n in names))
    print("-" * (width + 2 + 14 * len(names)))

    bad = refused = 0
    for p in args.probes:
        run.want = subprocess.run([CPY, str(p)], capture_output=True,
                                  text=True).stdout
        cells, outcomes = [], []
        for name in names:
            seen = {run(lyc, p, REGIMES[name]) for _ in range(args.runs)}
            cell = "ok" if seen == {"ok"} else "/".join(sorted(seen))
            outcomes.append(seen)
            cells.append(cell)
        # A probe lyc REFUSES in every regime never ran, so there is no
        # allocator behaviour for its outcome to depend on and this tool has
        # nothing to say about it. Counting it would make the exit code
        # unusable as a gate the moment the corpus holds a loud probe -- it
        # holds fifty -- and a checker that reports a healthy tree as red
        # teaches the reader to skim its output.
        if all(s == {"reject"} for s in outcomes):
            refused += 1
            flag = "   (refused, not an allocator finding)"
        elif any(s != {"ok"} for s in outcomes):
            bad += 1
            flag = "   <-- FAILS"
        else:
            flag = ""
        print(f"{p.name:{width}}  " + "  ".join(f"{c:>12}" for c in cells) + flag,
              flush=True)

    print(f"\n{bad}/{len(args.probes)} probes fail in at least one regime "
          f"({args.runs} runs per regime, {len(names)} regimes)")
    if refused:
        print(f"{refused} refused in every regime and are not counted: a "
              f"program that does not run cannot have an allocator-dependent "
              f"outcome")
    return bad


if __name__ == "__main__":
    sys.exit(main())
