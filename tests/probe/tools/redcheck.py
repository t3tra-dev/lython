#!/usr/bin/env python3
"""Has each golden ever been seen to FAIL?

A test is not validated by passing. It is validated by going red when the defect
it was written for is present, and a test that has only ever been green might be
green for the right reasons on the wrong code. k-4a hit exactly that: a
regression test for a `lowerInit` defect written with `@dataclass`, which
synthesises an `__init__` and therefore never reaches `lowerInit` at all -- green
throughout, and never once observed red.

This runs each golden against a binary from before the fix it was written for,
using history as the fault injector, and reports which ones can go red.

    python3 tests/probe/tools/redcheck.py ./build-pre/bin/lyc tests/golden/cases \
        --sentinel path/to/known_broken.py -- case_a case_b

--sentinel is required and is the reason this tool can be trusted. A GREEN result
means "this test cannot catch the defect", which is a serious claim, and it is
also what you get from an old binary that does not actually contain the defect --
a build pointed at the wrong tree, or one whose source changed underneath it. The
sentinel is a program known to fail on the old binary: if it passes, the binary is
not what it is supposed to be and every GREEN below it is meaningless, so the run
aborts instead of reporting them.

That check is not hypothetical. The first time I ran this by hand I started the
old build in the background and switched branches while it was still compiling,
so it linked a mixture of both trees and reported all three goldens as unable to
fail. Only knowing that one of them HAD failed historically caught it.

Exit code is the number of goldens that could not be made to fail.
"""

import argparse
import pathlib
import subprocess
import sys


def run(binary, path, timeout=900.0):
    try:
        r = subprocess.run([str(binary), "jit", str(path)], capture_output=True,
                           text=True, timeout=timeout)
        return r.returncode, r.stdout, r.stderr
    except subprocess.TimeoutExpired:
        return "timeout", "", ""


def why(rc, out, err):
    lines = [l for l in (out + "\n" + err).splitlines()
             if "refcount" in l or "error:" in l or "malloc" in l]
    return lines[0][:70] if lines else f"rc={rc}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("old_binary", type=pathlib.Path,
                    help="a build from before the fix under test")
    ap.add_argument("cases", type=pathlib.Path,
                    help="directory holding <name>.py and <name>.stdout")
    ap.add_argument("--sentinel", required=True, type=pathlib.Path,
                    help="a program KNOWN to fail on old_binary; if it passes, "
                         "the binary is wrong and no GREEN below is meaningful")
    ap.add_argument("names", nargs="+")
    args = ap.parse_args()
    old = args.old_binary.resolve()

    rc, out, err = run(old, args.sentinel)
    if rc == 0:
        print(f"SENTINEL PASSED on {old}", file=sys.stderr)
        print("The old binary does not exhibit the defect it is supposed to, so "
              "it cannot tell you whether a test catches it. Check that the "
              "build really came from the pre-fix tree and that nothing changed "
              "the source while it was compiling.", file=sys.stderr)
        return 2
    print(f"sentinel fails as required ({why(rc, out, err)}) -- binary usable\n")

    green = []
    for name in args.names:
        py = args.cases / f"{name}.py"
        expect_path = args.cases / f"{name}.stdout"
        expect = expect_path.read_text() if expect_path.exists() else None
        rc, out, err = run(old, py)
        if rc != 0:
            verdict = f"RED   rc={rc}  {why(rc, out, err)}"
        elif expect is not None and out != expect:
            verdict = f"RED   wrong output ({out.strip()[:40]!r})"
        else:
            verdict = "GREEN <-- cannot be made to fail; what does it cover?"
            green.append(name)
        print(f"{name:34} {verdict}", flush=True)

    print(f"\n{len(green)}/{len(args.names)} could not be made to fail")
    for n in green:
        print("  never red:", n)
    return len(green)


if __name__ == "__main__":
    sys.exit(main())
