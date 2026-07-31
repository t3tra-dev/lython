"""ctest runner for one lyc golden case.

Runs `lyc jit <case.py>` and verifies against sidecar files next to the case:
  <case>.stdout    expected stdout, exact match (optional)
  <case>.exitcode  expected exit code, exact match (optional, default 0)
  <case>.stderr-re regex that must match somewhere in stderr (optional)

--exit-only N skips sidecar lookup and only checks the exit code; ctest uses
it to smoke-run examples/ without adding expectation files there.

--aot builds an executable and runs it instead of JIT-ing, and --release passes
`--release` to lyc. Both are checked against the SAME sidecars by the SAME code
below: what they pin is that the other output mode and the release
configuration agree with the one the suite already believes. Reimplementing the
contract per mode is how they would drift, and the drift is not hypothetical --
a `def main()` was unbuildable as an executable while passing under JIT,
because nothing but the leak gate ever linked one.

--aot needs a scratch directory of its own: lyc writes the executable where it
is told but a case may write files into the working directory, and two of those
running under -j8 in one directory is a race.

--timeout S bounds the lyc run; exceeding it is reported as its own failure
reason rather than as differing output.

--expect-layer L asserts which pipeline stage the case reaches. The stage comes
from the compiler's own PerfScope trace rather than from the diagnostic text, so
a case that stops being rejected by the frontend and starts being rejected by
lowering -- or starts executing -- fails here by name instead of silently
getting more expensive. Only stages that never execute are declarable: the
trace adds lines to stderr, and stripping them back out is only provably
lossless while the program itself writes nothing.

A signal death is reported as a negative exit code and never satisfies an
expected exit code, so "must fail with exit 1" cannot be faked by a crash.

On any failure the runner also reports which stage the compiler reached, from a
second run under LYTHON_PERF=1. The report says so: it is not the run whose
streams were compared, and a nondeterministic case can disagree with itself.
"""

import argparse
import os
import pathlib
import re
import subprocess
import sys
import tempfile

PERF_LINE = re.compile(r"^\[LYTHON_PERF\] phase=(\S+)")

# Stages a case may be declared to stop at, ordered by depth. "e2e" is the
# default and is deliberately not declarable: it is what every case that runs
# the program reaches, so declaring it would assert nothing.
DECLARABLE_LAYERS = ("parse", "emit", "lower")


def phase_trace(stderr: str) -> "list[str]":
    """Phase names in the order the compiler printed them.

    PerfScope prints on scope exit, so an enclosing phase appears after the
    nested ones it contains, and a phase that failed still prints. The trace
    therefore ends with the outermost scope entered, and the deepest (dotted)
    name near the end is the one that rejected.
    """
    return [match.group(1)
            for match in (PERF_LINE.match(line)
                          for line in stderr.splitlines())
            if match]


def layer_of(phases: "list[str]") -> str:
    """The deepest pipeline stage a phase trace shows the compiler reaching."""
    if "execution" in phases:
        return "e2e"
    if any(phase.split(".", 1)[0] == "jit-build" for phase in phases):
        return "jit-build"
    if any(phase.split(".", 1)[0] == "lowering" for phase in phases):
        return "lower"
    if "ir-generation" in phases:
        return "emit"
    if "parse" in phases:
        return "parse"
    return "startup"


def strip_perf(stderr: str) -> str:
    kept = [line for line in stderr.splitlines() if not PERF_LINE.match(line)]
    return "".join(line + "\n" for line in kept)


def run_aot(lyc: pathlib.Path, case: pathlib.Path, timeout: float,
            env: "dict[str, str]", release: bool
            ) -> "subprocess.CompletedProcess[str] | None":
    """Build an executable, then run it. Failure to BUILD is returned as the
    result, so the caller reports it as this case failing rather than as a
    missing measurement -- the shape that let an unbuildable `def main()` sit in
    the suite while the leak gate skipped it."""
    with tempfile.TemporaryDirectory() as scratch:
        binary = pathlib.Path(scratch) / "prog"
        command = [str(lyc), str(case)]
        if release:
            command.append("--release")
        command += ["-o", str(binary)]
        try:
            built = subprocess.run(command, capture_output=True, text=True,
                                   timeout=timeout, env=env,
                                   stdin=subprocess.DEVNULL, cwd=scratch)
        except subprocess.TimeoutExpired:
            return None
        if built.returncode != 0:
            return built
        try:
            return subprocess.run([str(binary)], capture_output=True,
                                  text=True, timeout=timeout, env=env,
                                  stdin=subprocess.DEVNULL, cwd=scratch)
        except subprocess.TimeoutExpired:
            return None


def run_lyc(lyc: pathlib.Path, case: pathlib.Path, timeout: float,
            perf: bool, aot: bool = False, release: bool = False
            ) -> "subprocess.CompletedProcess[str] | None":
    env = dict(os.environ)
    if perf:
        env["LYTHON_PERF"] = "1"
    else:
        env.pop("LYTHON_PERF", None)
    if aot:
        return run_aot(lyc, case, timeout, env, release)
    try:
        # stdin=DEVNULL, not inherited: a case calling input() blocks until its
        # stdin reaches EOF, and whether the ambient stdin ever does is a
        # property of how ctest itself was launched -- not of the case. That is
        # how builtins_misc_wave15 timed out at 300 s in five full-suite runs
        # while passing standalone in 2.2 s at HIGHER load, which sent three
        # separate investigations after contention, build type and thread
        # starvation. DEVNULL is at EOF immediately, so input() raises EOFError
        # deterministically, which is what such a case is pinning anyway.
        command = [str(lyc), "jit"]
        if release:
            command.append("--release")
        command.append(str(case))
        return subprocess.run(command, capture_output=True, text=True,
                              timeout=timeout, env=env,
                              stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        return None


def first_difference(actual: str, expected: str) -> str:
    """Name the first differing line, so a 40-line expectation localizes."""
    actual_lines = actual.splitlines()
    expected_lines = expected.splitlines()
    for index in range(max(len(actual_lines), len(expected_lines))):
        got = actual_lines[index] if index < len(actual_lines) else "<no line>"
        want = (expected_lines[index] if index < len(expected_lines)
                else "<no line>")
        if got != want:
            return (f"first difference at stdout line {index + 1} "
                    f"(expected {len(expected_lines)} lines, "
                    f"got {len(actual_lines)}):\n"
                    f"  expected: {want!r}\n"
                    f"  actual:   {got!r}\n")
    return (f"every line matches; the streams differ in trailing bytes "
            f"(expected {len(expected)} bytes, actual {len(actual)})\n")


def fail(message: str, stdout: str, stderr: str, where: str = "") -> int:
    print(f"FAIL: {message}", file=sys.stderr)
    if where:
        sys.stderr.write(where)
    print("--- stdout ---", file=sys.stderr)
    sys.stderr.write(stdout)
    print("--- stderr ---", file=sys.stderr)
    sys.stderr.write(stderr)
    return 1


def report_reached_layer(lyc: pathlib.Path, case: pathlib.Path, timeout: float,
                         aot: bool = False, release: bool = False) -> None:
    """Say which stage the compiler reached, so a red test localizes itself.

    The re-run repeats the MODE as well as the case: a JIT re-run of an --aot
    failure would report a stage the failing run never went through.
    """
    result = run_lyc(lyc, case, timeout, perf=True, aot=aot, release=release)
    if result is None:
        print("--- reached layer: unknown, the LYTHON_PERF re-run timed out",
              file=sys.stderr)
        return
    phases = phase_trace(result.stderr)
    print(f"--- reached layer, from a SECOND run under LYTHON_PERF=1 and not "
          f"the run compared above: {layer_of(phases)}", file=sys.stderr)
    if not phases:
        print("--- no phase printed: lyc exited before parsing",
              file=sys.stderr)
        return
    dotted = [phase for phase in phases if "." in phase]
    if dotted:
        print(f"--- deepest phase entered: {dotted[-1]}", file=sys.stderr)
    print(f"--- last phases printed: {' '.join(phases[-5:])}", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lyc", required=True, type=pathlib.Path)
    parser.add_argument("--exit-only", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--expect-layer", choices=DECLARABLE_LAYERS,
                        default=None)
    parser.add_argument("--aot", action="store_true")
    parser.add_argument("--release", action="store_true")
    parser.add_argument("case", type=pathlib.Path)
    args = parser.parse_args()

    # Why not let TimeoutExpired propagate: an uncaught traceback exits
    # nonzero, so ctest labels the run "Failed" exactly like a wrong-output
    # case and the report gives no hint that the budget was the cause.
    result = run_lyc(args.lyc, args.case, args.timeout,
                     perf=args.expect_layer is not None, aot=args.aot,
                     release=args.release)
    if result is None:
        # Why no layer report here: the re-run would spend the same budget
        # over again and end the same way.
        print(f"FAIL: lyc did not finish within {args.timeout:g}s",
              file=sys.stderr)
        return 1

    stdout = result.stdout
    stderr = result.stderr
    reached = None
    if args.expect_layer is not None:
        reached = layer_of(phase_trace(stderr))
        stderr = strip_perf(stderr)

    def failed(message: str, where: str = "") -> int:
        code = fail(message, stdout, stderr, where)
        if args.expect_layer is None:
            report_reached_layer(args.lyc, args.case, args.timeout,
                                 aot=args.aot, release=args.release)
        else:
            print(f"--- reached layer: {reached}", file=sys.stderr)
        return code

    if args.exit_only is not None:
        if result.returncode != args.exit_only:
            return failed(f"exit code {result.returncode}, "
                          f"expected {args.exit_only}")
        return 0

    expected_exit = 0
    exitcode_file = args.case.with_suffix(".exitcode")
    if exitcode_file.exists():
        expected_exit = int(exitcode_file.read_text().strip())
    if result.returncode != expected_exit:
        return failed(f"exit code {result.returncode}, "
                      f"expected {expected_exit}")

    stdout_file = args.case.with_suffix(".stdout")
    if stdout_file.exists():
        expected_stdout = stdout_file.read_text()
        if stdout != expected_stdout:
            return failed("stdout differs from expected",
                          first_difference(stdout, expected_stdout))

    stderr_re_file = args.case.with_suffix(".stderr-re")
    if stderr_re_file.exists():
        pattern = stderr_re_file.read_text().strip()
        if not re.search(pattern, stderr):
            return failed(f"stderr does not match /{pattern}/")

    # Checked last: the expectations above are what the case is for, and a
    # layer that drifted while every expectation still holds is a cost
    # regression rather than a wrong answer. Reporting it last keeps the two
    # apart in the ctest output.
    if args.expect_layer is not None and reached != args.expect_layer:
        return fail(
            f"case reached layer {reached!r} but tests/golden/layers.txt "
            f"declares {args.expect_layer!r}",
            stdout, stderr,
            "Every expectation still holds; what changed is how far the "
            "compiler runs. Either the declaration is stale (update "
            "tests/golden/layers.txt) or a stage stopped rejecting this "
            "program and a later, more expensive one now does.\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
