#!/usr/bin/env python3
"""Fail if a program leaks more than the runtime's own fixed baseline.

Why this exists: nothing in the suite could see a leak. Goldens compare exit
code and stdout, so a leaking program is indistinguishable from a correct one --
five goldens were green while leaking, and one defect leaked 64 bytes per
iteration without bound. tests/probe/tools/leak.py cannot cover the gap either:
it watches RSS growth per iteration and its floor is 500 B/iter, so it is blind
both to the bounded classes and to that 64 B/iter unbounded one.

⭐ MEASURED BY LeakSanitizer, not by `leaks`. The first version of this gate ran
`leaks --atExit`, which is macOS-only, so the stage registered NOTHING on Linux --
including in CI. `lyc --fsanitize=leak` measures the same thing on both, and the
two were checked against each other on all twelve members of the stage (identical
figures) and on a deliberately broken compiler (LSan 78 allocations, `leaks` 78).

Two things about LSan had to be measured rather than assumed, and both are why the
options below are hardcoded instead of left to the environment:

  * IT IS NONDETERMINISTIC BY DEFAULT. Over twelve runs, a clean program reported
    a leak 9 times and a leaking one flipped between two different byte totals,
    because the conservative root scan sometimes finds a stale pointer on the
    stack or in a register and calls the allocation reachable.
    `use_stacks=0:use_registers=0` makes it exactly deterministic -- 8/8 identical
    on three programs, including a real member of the stage.
  * IT REPORTS THE BASELINE TOO. Every AOT binary carries one root of ~524 KB from
    LyRt_InstallStackGuard, and LSan reports it like `leaks` did, so the
    subtraction below is still the whole point. A coordinator once read the
    absolute figure as a program leak.
  * ⛔ AND IT STILL GOES SILENT, RARELY. The options above do NOT make it fully
    deterministic: measured over 40 runs each, one clean case printed nothing at
    all 1 time in 40 while the baseline and a leaking binary printed every time.
    The FIGURE is deterministic -- every run that reported gave the identical
    number -- so the only failure is occasional total silence. That is why
    `measure` retries: silence carries no information, and the first version of
    this file claimed a summary was always produced, which cost a red run to
    disprove.

    Retrying is sound in the direction that matters. Silence never becomes a
    pass: it becomes either a later measurement or a refusal, so a leaking
    program can be reported correctly or skipped but not called clean. Three
    attempts, from the measured 1-in-40: three consecutive silences is about
    1.6e-5, against roughly 4e-4 per full suite run.

What it measures: the LSan total on an AOT binary, MINUS the baseline measured on
`print(0)` in the same run, never assumed.

Three ways it refuses instead of answering, because the thing being looked for is
an absence and an absence is what a broken instrument also reports:

  1. The program must exit with the code it is DECLARED to exit with, checked
     BEFORE going near the leak detector. LSan exits 23 whenever it finds
     anything -- including the baseline -- so its status masks the program's,
     exactly as `leaks` did (a program exiting 3 came back as 1). The check runs
     the same binary under `detect_leaks=0`, which gives the program's own exit
     code back. A crash read through an unguarded parser is a clean zero; that is
     how a shipped SIGSEGV was once relayed as "this case leaks 0".

     The declaration is `--expect-exit N`, default 0. Requiring 0 outright is
     what it used to do, and it put EVERY program that ends by design with a
     non-zero status outside what this gate can see -- `sys.exit()` and every
     uncaught exception. That class was not empty: the reference
     `LyEH_TakeCurrentDescriptor` transfers to `LyRunPythonMain` was never
     discharged, so all of them leaked the exception object. Declaring the
     expected code keeps the guard (silence and crashes are still refused) and
     stops it from being a blind spot.
  2. The baseline run must produce a parseable summary within ATTEMPTS tries.
     Without one there is no zero point and every number below is meaningless.
  3. The subject run must produce one too. Silence is not zero -- treating it as
     zero is exactly how a leaking program would come back green.

Exit 0 = net zero. 1 = leaked. 2 = could not measure (refusal). ctest maps 2 to
SKIP, so a toolchain without a leak sanitizer skips the stage rather than failing
it -- and rather than silently registering a green it never measured.

⭐ THE BASELINE IS BUILT ONCE PER RUN. `--build-baseline DIR` (a ctest fixture
setup) builds and measures `print(0)` and records the figure, and the compiler
it came from, in DIR/baseline.json; each case reads it with `--baseline FILE`.
Every case used to build its own -- the same sanitized binary 94 times, about
half of the stage. A recorded baseline from a different lyc (path, size,
modification time, extra flags) is refused rather than used: a stale zero point
would make every figure below it wrong without saying so.
"""

import argparse
import json
import os
import pathlib
import re
import subprocess
import sys
import tempfile

SUMMARY = re.compile(
    r"SUMMARY: \w*Sanitizer: (\d+) byte\(s\) leaked in (\d+) allocation")

# Allocations made by system libraries on our behalf and never freed by them.
# Kept in a file next to this one rather than inline so the reasoning for each
# entry has room; see the file. Without it the two `prim_tensor_*` cases report
# 208 B of libdispatch's one-time dispatch_apply cache, which reads exactly like
# a compiler defect and was chased as one.
SUPPRESSIONS = pathlib.Path(__file__).with_name("lsan_suppressions.txt")

# Hardcoded, not merged with the caller's environment: the determinism of every
# figure this gate prints depends on them, so letting LSAN_OPTIONS through from
# outside would let a caller silently turn the gate into a coin flip.
DETECT_ENV = f"use_stacks=0:use_registers=0:suppressions={SUPPRESSIONS}"
NO_DETECT_ENV = "detect_leaks=0"

# From the measured 1-in-40 silence rate; see the module docstring.
ATTEMPTS = 3

# Sensitivity checks need a compiler whose verifier phases are off, because the
# owned-token uniqueness gate refuses the very IR the ablation produces.
EXTRA_LYC_FLAGS = os.environ.get("LYTHON_LEAK_GATE_LYC_FLAGS", "").split()


def child_env(lsan_options: str) -> "dict[str, str]":
    env = dict(os.environ)
    env["LSAN_OPTIONS"] = lsan_options
    return env


def run_alone(binary: pathlib.Path, timeout: float) -> int:
    """The program's OWN exit code, with leak detection switched off."""
    try:
        r = subprocess.run([str(binary)], capture_output=True, text=True,
                           timeout=timeout, stdin=subprocess.DEVNULL,
                           env=child_env(NO_DETECT_ENV))
        return r.returncode
    except subprocess.TimeoutExpired:
        return -1


def measure_once(binary: pathlib.Path,
                 timeout: float) -> "tuple[int, int] | str":
    """(bytes, allocations) as LSan reports them, or a REASON string.

    A reason rather than None, because "it timed out" and "it ran and said
    nothing" are different failures and the first version of this collapsed them
    into one message -- which sent a diagnosis looking for a missing summary when
    the run had in fact died.
    """
    try:
        r = subprocess.run([str(binary)], capture_output=True, text=True,
                           timeout=timeout, stdin=subprocess.DEVNULL,
                           env=child_env(DETECT_ENV))
    except subprocess.TimeoutExpired:
        return f"timed out after {timeout}s"
    # LSan's own exit status is deliberately ignored: it is 23 whenever it finds
    # anything at all, including the baseline every binary carries.
    match = SUMMARY.search(r.stdout + "\n" + r.stderr)
    if match:
        return (int(match.group(1)), int(match.group(2)))
    tail = (r.stderr or r.stdout).strip().splitlines()[-3:]
    return (f"ran with rc={r.returncode} and printed no summary; last lines: "
            + " | ".join(tail))


def measure(binary: pathlib.Path,
            timeout: float) -> "tuple[int, int] | str":
    """The first attempt that reports, or the last reason if none do.

    Repeats because LSan goes silent about 1 run in 40 (measured) while the number
    it prints when it does report is identical every time. A retry therefore only
    ever turns "no measurement" into "a measurement"; it cannot turn one figure
    into another.
    """
    reason = "not attempted"
    for _ in range(ATTEMPTS):
        result = measure_once(binary, timeout)
        if not isinstance(result, str):
            return result
        reason = result
        if reason.startswith("timed out"):
            return reason  # not a flake: repeating costs the timeout again
    return f"silent on all {ATTEMPTS} attempts ({reason})"


def build(lyc: pathlib.Path, source: pathlib.Path, out: pathlib.Path,
          timeout: float) -> bool:
    """Build in `out`'s directory, not the caller's.

    ⛔ The docstring at the top of this file has claimed "a scratch cwd" since the
    gate was written, and the code never changed directory. `lyc` drops an `a.out`
    beside the working directory, so a stale one from any earlier run makes every
    later build fail with `symbol 'main' already exists` -- reported as "could not
    measure", which after SKIP_RETURN_CODE reads as a skip rather than a problem.
    It cost a whole-suite survey two unmeasured cases and a diagnosis session an
    afternoon. Now the claim is true.
    """
    try:
        r = subprocess.run([str(lyc), str(source), "--fsanitize=leak",
                            *EXTRA_LYC_FLAGS, "-o", str(out)],
                           capture_output=True, text=True, timeout=timeout,
                           stdin=subprocess.DEVNULL, cwd=str(out.parent))
    except subprocess.TimeoutExpired:
        return False
    if r.returncode != 0:
        print(f"sanitized AOT build failed: {r.stderr.strip()[:300]}",
              file=sys.stderr)
    return r.returncode == 0


def compiler_fingerprint(lyc: pathlib.Path) -> "dict[str, object]":
    stat = lyc.stat()
    return {"lyc": str(lyc), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            "flags": EXTRA_LYC_FLAGS}


def build_baseline(lyc: pathlib.Path, work: pathlib.Path,
                   timeout: float) -> "tuple[int, int] | str":
    """Build and measure `print(0)`: the figure or why there is none."""
    baseline_src = work / "_baseline.py"
    baseline_src.write_text("print(0)\n")
    baseline_bin = work / "baseline"
    if not build(lyc, baseline_src, baseline_bin, timeout):
        return ("could not build a leak-sanitized binary for `print(0)`; this "
                "toolchain cannot measure leaks")
    code = run_alone(baseline_bin, timeout)
    if code != 0:
        return (f"the baseline exits {code} on its own, not 0; the sanitizer's "
                f"own status would mask that")
    base = measure(baseline_bin, timeout)
    if isinstance(base, str):
        return ("no leak summary for the baseline: with detection on, the "
                "stack-guard root is always found, so a missing summary means "
                f"the instrument is not running ({base})")
    return base


def write_baseline(lyc: pathlib.Path, out_dir: pathlib.Path,
                   timeout: float) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as scratch:
        base = build_baseline(lyc, pathlib.Path(scratch), timeout)
    record = compiler_fingerprint(lyc)
    if isinstance(base, str):
        # ⛔ Recorded, not raised: the cases read it and skip with the reason,
        # as each did when it built its own.
        record["unmeasurable"] = base
    else:
        record["bytes"], record["allocs"] = base
    (out_dir / "baseline.json").write_text(json.dumps(record))
    print(f"baseline: {record}")
    return 0


def read_baseline(path: pathlib.Path,
                  lyc: pathlib.Path) -> "tuple[int, int] | str":
    if not path.exists():
        return f"no recorded baseline at {path}"
    record = json.loads(path.read_text())
    expected = compiler_fingerprint(lyc)
    for key, value in expected.items():
        if record.get(key) != value:
            return (f"the recorded baseline came from another compiler "
                    f"({key}: {record.get(key)!r}, now {value!r})")
    if "unmeasurable" in record:
        return str(record["unmeasurable"])
    return int(record["bytes"]), int(record["allocs"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("lyc", type=pathlib.Path)
    ap.add_argument("source", type=pathlib.Path, nargs="?")
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--build-baseline", type=pathlib.Path, default=None,
                    help="build and record the baseline in this directory")
    ap.add_argument("--baseline", type=pathlib.Path, default=None,
                    help="the recorded baseline.json to measure against")
    # The subject's own exit code. The BASELINE is always 0: it is `print(0)`.
    ap.add_argument("--expect-exit", type=int, default=0)
    args = ap.parse_args()

    lyc = args.lyc.resolve()
    if args.build_baseline is not None:
        return write_baseline(lyc, args.build_baseline, args.timeout)
    if args.source is None:
        print("a source to measure is required", file=sys.stderr)
        return 2
    source = args.source.resolve()
    # The suppressions file is checked like the binary and the source, not
    # assumed: LSan aborts the child when it cannot open one, which arrives here
    # as silence and would be reported as "could not measure" -- true, but
    # pointing at the sanitizer instead of at the missing file.
    for label, path in (("lyc", lyc), ("source", source),
                        ("suppressions file", SUPPRESSIONS)):
        if not path.exists():
            print(f"{label} does not exist: {path}", file=sys.stderr)
            return 2

    # A scratch cwd: lyc drops a.out into the working directory, and a stale one
    # makes the NEXT program fail to link with "symbol 'main' already exists".
    with tempfile.TemporaryDirectory() as scratch:
        work = pathlib.Path(scratch)
        subject_bin = work / "subject"
        # The baseline FIRST: it is `print(0)`, so if the toolchain has no leak
        # sanitizer at all this fails on the cheapest possible program and the
        # stage skips, instead of reporting a subject-specific failure for a
        # missing runtime.
        base = (read_baseline(args.baseline, lyc) if args.baseline is not None
                else build_baseline(lyc, work, args.timeout))
        if isinstance(base, str):
            print(f"{base}. Refusing.", file=sys.stderr)
            return 2
        if not build(lyc, source, subject_bin, args.timeout):
            return 2
        code = run_alone(subject_bin, args.timeout)
        if code != args.expect_exit:
            print(f"subject exits {code} on its own, not the declared "
                  f"{args.expect_exit}; the sanitizer's own status would mask "
                  f"that, so a measurement here could read as zero. Refusing.",
                  file=sys.stderr)
            return 2
        subject = measure(subject_bin, args.timeout)
        if isinstance(subject, str):
            print(f"no leak summary for the subject. Silence is not zero "
                  f"({subject}). Refusing.", file=sys.stderr)
            return 2

        net_bytes = subject[0] - base[0]
        net_allocs = subject[1] - base[1]
        print(f"{source.name}: subject {subject[1]}/{subject[0]} B  "
              f"baseline {base[1]}/{base[0]} B  "
              f"net {net_allocs} allocs / {net_bytes} B")
        if net_allocs > 0 or net_bytes > 0:
            print(f"LEAK: {net_allocs} allocations / {net_bytes} bytes above "
                  f"baseline", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
