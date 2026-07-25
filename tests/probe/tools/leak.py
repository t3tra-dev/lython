#!/usr/bin/env python3
"""Measure bytes retained per iteration for each leak probe.

Absolute peak RSS says nothing here: `lyc jit` compiles the module in the same
process that runs it, and the compiler dominates the footprint. So each leak
probe ships as two spellings of one program, `leak_<key>_small.py` and
`leak_<key>_big.py`, differing only in the iteration count. Both do identical
compiler work, so the difference in peak RSS divided by the difference in
iterations is the bytes the loop failed to release.

    python3 tests/probe/tools/leak.py ./build/bin/lyc tests/probe [out.json]

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
    for key in keys:
        big = args.probes / f"leak_{key}_big.py"
        if not big.exists():
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
    if args.out:
        args.out.write_text(json.dumps(rows, indent=1))
    return leaking


if __name__ == "__main__":
    sys.exit(main())
