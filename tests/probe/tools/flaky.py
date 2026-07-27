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

"Nothing was unstable" is a negative result, so it cannot be told apart from
"nothing was observed" without an input that IS unstable. Until stage 4b that
input was `alias_read_mutate_nowriteback_dict`, which was in the corpus by
accident of not being fixed yet; 4b fixed it, and this tool would have gone on
reporting stability against an input set it could no longer be wrong about.

    python3 tests/probe/tools/flaky.py ./build/bin/lyc --self-test

runs `fixtures/unstable_by_construction.py`, whose output varies between runs of
the same binary for a reason no stage can repair, and exits non-zero unless
`NOT STABLE` appears. That covers instability of the OUTPUT; it does not rehearse
the face distribution of a real use-after-free (ok / silent / abort / signal),
which still needs a broken tree.
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
        r = subprocess.run(cmd,
                           capture_output=True, stdin=subprocess.DEVNULL,
                           text=True, timeout=timeout,
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


FIXTURE = (pathlib.Path(__file__).parent / "fixtures" /
           "unstable_by_construction.py")


def self_test(lyc, runs):
    """Show that this tool can say NOT STABLE at all, on this binary, today."""
    want = run([CPY, str(FIXTURE)])[1]
    faces = Counter()
    for _ in range(runs):
        faces[face(*run([lyc.as_posix(), "jit", str(FIXTURE)]), want=want)] += 1
    print(f"{FIXTURE.name}  plain: {dict(faces)}")
    if len(faces) > 1:
        print("\nself-test PASSED: the instability was reported")
        return 0
    print("\nself-test FAILED: an input that varies between runs was reported "
          "as stable, so a stable verdict from this tool means nothing today")
    return 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("lyc", type=pathlib.Path)
    ap.add_argument("probes", nargs="*", type=pathlib.Path)
    ap.add_argument("-n", "--runs", type=int, default=8)
    ap.add_argument("--self-test", action="store_true",
                    help="check that this tool can report instability at all")
    args = ap.parse_args()
    lyc = args.lyc.resolve()

    if args.self_test:
        return self_test(lyc, max(args.runs, 4))
    if not args.probes:
        ap.error("give probes, or --self-test")

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
