#!/usr/bin/env python3
"""Measure bytes retained per iteration for each leak probe.

Absolute peak RSS says nothing here: `lyc jit` compiles the module in the same
process that runs it, and the compiler dominates the footprint. So each leak
probe ships as two spellings of one program, `leak_<key>_small.py` and
`leak_<key>_big.py`, differing only in the iteration count. Both do identical
compiler work, so the difference in peak RSS divided by the difference in
iterations is the bytes the loop failed to release.

    python3 tests/probe/tools/leak.py ./build/bin/lyc tests/probe [out.json]

VALIDATING THIS INSTRUMENT. Its negative result -- "no leak" -- is the healthy
state, so a tree with no leaks cannot demonstrate that it is capable of saying
LEAK at all. It can only be domain-tested against a tree that leaks, which means
the recipe has to outlive any particular build directory:

    git worktree/checkout 657f0d8 ; cmake -B build-pre -S . -G Ninja
    cmake --build build-pre -j$(nproc)          # let it FINISH before switching
    python3 tests/probe/tools/leak.py ./build-pre/bin/lyc tests/probe

Expected there: `rebind_call_list` about 8500 B/iteration and
`rebind_twice_call_list` about 17200, reported as `2/12 shapes leak`. Anything
that reports 0/12 on that commit is not measuring what it claims. Measured
values were 8499 and 17191, against 8438 and 17176 recorded in the facts table
for the same commit -- agreement within run-to-run spread.

Note the "let it finish" above: building in the background and switching branches
underneath it links a mixture of two trees and produces a binary that silently
does not contain the defect. That mistake has been made here once already.

The floor is calibrated against the probes that must not leak. `leak_baseline_*`
are plain create-and-drop loops, and together with the non-leaking rebind shapes
they have been observed anywhere in -130..+130 bytes per iteration across runs:
one sample of peak RSS is a coarse instrument at the small end, because the
allocator's own high-water mark moves with page-level rounding that has nothing
to do with the loop. The floor is set well above that spread rather than at its
edge, since a real leak in this family is 8500+ bytes per iteration -- more than
an order of magnitude clear of it -- so a generous floor costs no sensitivity
and stops a noisy sample from being reported as a leak.
"""

import argparse
import json
import pathlib
import re
import subprocess
import sys

# Kept in sync with the iteration counts baked into the probe pairs.
SMALL, BIG = 100, 40000
NOISE = 500.0

RSS_RE = re.compile(r"^\s*(\d+)\s+maximum resident set size", re.M)


def peak_rss(lyc, case):
    r = subprocess.run(["/usr/bin/time", "-l", str(lyc), "jit", str(case)],
                       capture_output=True, text=True, timeout=1800)
    m = RSS_RE.search(r.stderr)
    return (int(m.group(1)) if m else None), r.returncode, r.stdout.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("lyc", type=pathlib.Path)
    ap.add_argument("probes", type=pathlib.Path)
    ap.add_argument("out", nargs="?", type=pathlib.Path, default=None)
    args = ap.parse_args()
    lyc = args.lyc.resolve()

    keys = sorted({p.name[len("leak_"):-len("_small.py")]
                   for p in args.probes.glob("leak_*_small.py")})
    rows = {}
    leaking = 0
    unpaired = []
    for key in keys:
        big = args.probes / f"leak_{key}_big.py"
        if not big.exists():
            # Not `continue`: silently dropping a probe makes the summary count
            # smaller with nothing to say why, which is the same silence this
            # file's own docstring warns about. An unpaired probe is a mistake
            # in the corpus, so it is reported and counted.
            print(f"{key:34} NO _big COUNTERPART -- cannot be measured",
                  flush=True)
            unpaired.append(key)
            continue
        s_rss, s_rc, s_out = peak_rss(lyc, args.probes / f"leak_{key}_small.py")
        b_rss, b_rc, b_out = peak_rss(lyc, big)
        per_iter = None
        if s_rss and b_rss:
            per_iter = (b_rss - s_rss) / (BIG - SMALL)
        verdict = "?" if per_iter is None else (
            "LEAK" if per_iter > NOISE else "no leak")
        if verdict == "LEAK":
            leaking += 1
        rows[key] = dict(small_rss=s_rss, big_rss=b_rss, small_rc=s_rc,
                         big_rc=b_rc, small_out=s_out, big_out=b_out,
                         bytes_per_iter=per_iter, verdict=verdict)
        print(f"{key:34} small={(s_rss or 0) // 1024:>7}K "
              f"big={(b_rss or 0) // 1024:>8}K "
              f"per_iter={'?' if per_iter is None else round(per_iter, 1):>9} "
              f"{verdict:8} rc={s_rc}/{b_rc}", flush=True)

    print(f"\n{leaking}/{len(rows)} shapes leak (floor {NOISE:g} B/iteration)")
    if unpaired:
        print(f"{len(unpaired)} unpaired and therefore unmeasured: "
              + ", ".join(unpaired))
    if args.out:
        args.out.write_text(json.dumps(rows, indent=1))
    # An unmeasurable probe counts against the run: the alternative is a clean
    # exit that quietly measured less than the corpus contains.
    return leaking + len(unpaired)


if __name__ == "__main__":
    sys.exit(main())
