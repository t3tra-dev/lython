#!/usr/bin/env python3
"""Decide whether a probe's outcome is stable, and get a verdict when it is not.

Some object-ownership shapes are use-after-frees whose visible face depends on
allocator state: the same binary on the same input prints the right answer on
some runs, prints a wrong one on others, and aborts on the rest. For those, a
single run's classification is noise, and comparing a single run before and
after a change can invent a regression or hide one.

This runs each probe N times plainly and once under libgmalloc, which places
allocations so that a freed block is unmapped rather than reused. Every shape in
the alias-read family dies under it on every run, and every shape that is
genuinely correct survives, so the guard-allocator column is the verdict and the
plain column only says how deceptive the shape is without it.

    python3 tests/probe/tools/flaky.py ./build/bin/lyc -n 10 tests/probe/alias_*.py

Exit code is the number of probes whose libgmalloc run did not match CPython.
"""

import argparse
import os
import pathlib
import re
import subprocess
import sys
from collections import Counter

CPY = "/opt/homebrew/Frameworks/Python.framework/Versions/3.14/bin/python3.14"
GM = dict(DYLD_INSERT_LIBRARIES="/usr/lib/libgmalloc.dylib",
          MALLOC_PROTECT_BEFORE="1", MALLOC_FILL_SPACE="1")
REFCOUNT = re.compile(r"Ly_(?:Inc|Dec)Ref observed non-positive refcount")


def run(cmd, env_extra=None, timeout=900.0):
    env = dict(os.environ)
    if env_extra:
        env.update(env_extra)
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                           env=env)
        return r.returncode, r.stdout, r.stderr
    except subprocess.TimeoutExpired:
        return "timeout", "", ""


def face(rc, out, err, want):
    """Name the outcome of one run, in the vocabulary of the facts table."""
    if rc == "timeout":
        return "timeout"
    if rc == 0:
        return "ok" if out == want else f"SILENT({out.strip()})"
    if REFCOUNT.search(out + err):
        return "abort(refcount)"
    if isinstance(rc, int) and rc < 0:
        return f"signal {-rc}"
    errs = [l for l in err.splitlines() if "error" in l.lower()]
    return "reject" if errs else f"exit {rc}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("lyc", type=pathlib.Path)
    ap.add_argument("probes", nargs="+", type=pathlib.Path)
    ap.add_argument("-n", "--runs", type=int, default=8)
    args = ap.parse_args()
    lyc = args.lyc.resolve()

    bad = 0
    width = max(len(p.name) for p in args.probes)
    for p in args.probes:
        want = run([CPY, str(p)])[1]
        faces = Counter()
        for _ in range(args.runs):
            faces[face(*run([lyc.as_posix(), "jit", str(p)]), want=want)] += 1
        gm = face(*run([lyc.as_posix(), "jit", str(p)], GM), want=want)
        if gm != "ok":
            bad += 1
        stable = "" if len(faces) == 1 else "  <- NOT STABLE"
        print(f"{p.name:{width}}  plain: {dict(faces)}   gmalloc: {gm}{stable}",
              flush=True)
    print(f"\n{bad}/{len(args.probes)} fail under libgmalloc")
    return bad


if __name__ == "__main__":
    sys.exit(main())
