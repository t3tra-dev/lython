#!/usr/bin/env python3
"""Fail if a program leaks more than the runtime's own fixed baseline.

Why this exists: nothing in the suite could see a leak. Goldens compare exit
code and stdout, so a leaking program is indistinguishable from a correct one --
five goldens were green while leaking, and one defect leaked 64 bytes per
iteration without bound. tests/probe/tools/leak.py cannot cover the gap either:
it watches RSS growth per iteration and its floor is 500 B/iter, so it is blind
both to the bounded classes and to that 64 B/iter unbounded one.

What it measures: `leaks --atExit` on an AOT binary, MINUS the baseline that
every AOT binary carries (one root of ~528 KB from LyRt_InstallStackGuard). The
baseline is measured on `print(0)` in the same run, never assumed -- a
coordinator read the absolute figure as a program leak once, and it was almost
entirely this.

Three ways it refuses instead of answering, because the thing being looked for
is an absence and an absence is what a broken instrument also reports:

  1. The program must exit 0 on its own, checked BEFORE going near `leaks`.
     `leaks` reports NOTHING for a process that crashed, and it also masks the
     program's own status (a program exiting 3 comes back as 1). A crash read
     through an unguarded parser is a clean zero -- that is how a shipped
     SIGSEGV was once relayed as "this case leaks 0".
  2. The baseline run must produce a parseable summary. Without it there is no
     zero point and every number below is meaningless.
  3. The subject run must produce a parseable summary too. Silence is not zero.

Exit 0 = net zero. 1 = leaked. 2 = could not measure (refusal).
"""

import argparse
import pathlib
import re
import subprocess
import sys
import tempfile

SUMMARY = re.compile(r"(\d+) leaks? for (\d+) total leaked bytes")


def run_alone(binary: pathlib.Path, timeout: float) -> "tuple[int, str]":
    try:
        r = subprocess.run([str(binary)], capture_output=True, text=True,
                           timeout=timeout, stdin=subprocess.DEVNULL)
        return r.returncode, r.stdout
    except subprocess.TimeoutExpired:
        return -1, ""


def measure(binary: pathlib.Path, timeout: float) -> "tuple[int, int] | None":
    """(roots, bytes) as `leaks` reports them, or None if it reported nothing."""
    try:
        r = subprocess.run(["leaks", "--atExit", "--", str(binary)],
                           capture_output=True, text=True, timeout=timeout,
                           stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        return None
    # leaks' own exit status is deliberately ignored: it is nonzero whenever it
    # finds anything at all, including the baseline every binary carries.
    match = SUMMARY.search(r.stdout + "\n" + r.stderr)
    return (int(match.group(1)), int(match.group(2))) if match else None


def build(lyc: pathlib.Path, source: pathlib.Path, out: pathlib.Path,
          timeout: float) -> bool:
    try:
        r = subprocess.run([str(lyc), str(source), "-o", str(out)],
                           capture_output=True, text=True, timeout=timeout,
                           stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        return False
    if r.returncode != 0:
        print(f"AOT build failed: {r.stderr.strip()[:200]}", file=sys.stderr)
    return r.returncode == 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("lyc", type=pathlib.Path)
    ap.add_argument("source", type=pathlib.Path)
    ap.add_argument("--timeout", type=float, default=300.0)
    args = ap.parse_args()

    lyc = args.lyc.resolve()
    source = args.source.resolve()
    for label, path in (("lyc", lyc), ("source", source)):
        if not path.exists():
            print(f"{label} does not exist: {path}", file=sys.stderr)
            return 2

    # A scratch cwd: lyc drops a.out into the working directory, and a stale one
    # makes the NEXT program fail to link with "symbol 'main' already exists".
    with tempfile.TemporaryDirectory() as scratch:
        work = pathlib.Path(scratch)
        baseline_src = work / "_baseline.py"
        baseline_src.write_text("print(0)\n")

        subject_bin = work / "subject"
        baseline_bin = work / "baseline"
        if not build(lyc, source, subject_bin, args.timeout):
            return 2
        if not build(lyc, baseline_src, baseline_bin, args.timeout):
            return 2

        for label, binary in (("subject", subject_bin),
                              ("baseline", baseline_bin)):
            code, _ = run_alone(binary, args.timeout)
            if code != 0:
                print(f"{label} does not exit 0 on its own (rc={code}); "
                      f"`leaks` reports nothing for a process that dies, so a "
                      f"measurement here would read as zero. Refusing.",
                      file=sys.stderr)
                return 2

        base = measure(baseline_bin, args.timeout)
        if base is None:
            print("no leaks summary for the baseline: without a zero point "
                  "every figure below is meaningless. Refusing.",
                  file=sys.stderr)
            return 2
        subject = measure(subject_bin, args.timeout)
        if subject is None:
            print("no leaks summary for the subject. Silence is not zero. "
                  "Refusing.", file=sys.stderr)
            return 2

        net_roots = subject[0] - base[0]
        net_bytes = subject[1] - base[1]
        print(f"{source.name}: subject {subject[0]}/{subject[1]} B  "
              f"baseline {base[0]}/{base[1]} B  "
              f"net {net_roots} roots / {net_bytes} B")
        if net_roots > 0 or net_bytes > 0:
            print(f"LEAK: {net_roots} roots / {net_bytes} bytes above baseline",
                  file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
