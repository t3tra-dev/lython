"""ctest runner for one lyc golden case.

Runs `lyc jit <case.py>` and verifies against sidecar files next to the case:
  <case>.stdout    expected stdout, exact match (optional)
  <case>.exitcode  expected exit code, exact match (optional, default 0)
  <case>.stderr-re regex that must match somewhere in stderr (optional)

--exit-only N skips sidecar lookup and only checks the exit code; ctest uses
it to smoke-run examples/ without adding expectation files there.

A signal death is reported as a negative exit code and never satisfies an
expected exit code, so "must fail with exit 1" cannot be faked by a crash.
"""

import argparse
import pathlib
import re
import subprocess
import sys


def fail(message: str, stdout: str, stderr: str) -> int:
    print(f"FAIL: {message}", file=sys.stderr)
    print("--- stdout ---", file=sys.stderr)
    sys.stderr.write(stdout)
    print("--- stderr ---", file=sys.stderr)
    sys.stderr.write(stderr)
    return 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lyc", required=True, type=pathlib.Path)
    parser.add_argument("--exit-only", type=int, default=None)
    parser.add_argument("case", type=pathlib.Path)
    args = parser.parse_args()

    result = subprocess.run(
        [str(args.lyc), "jit", str(args.case)],
        capture_output=True,
        text=True,
        timeout=300,
    )

    if args.exit_only is not None:
        if result.returncode != args.exit_only:
            return fail(
                f"exit code {result.returncode}, expected {args.exit_only}",
                result.stdout,
                result.stderr,
            )
        return 0

    expected_exit = 0
    exitcode_file = args.case.with_suffix(".exitcode")
    if exitcode_file.exists():
        expected_exit = int(exitcode_file.read_text().strip())
    if result.returncode != expected_exit:
        return fail(
            f"exit code {result.returncode}, expected {expected_exit}",
            result.stdout,
            result.stderr,
        )

    stdout_file = args.case.with_suffix(".stdout")
    if stdout_file.exists():
        expected_stdout = stdout_file.read_text()
        if result.stdout != expected_stdout:
            return fail("stdout differs from expected", result.stdout,
                        result.stderr)

    stderr_re_file = args.case.with_suffix(".stderr-re")
    if stderr_re_file.exists():
        pattern = stderr_re_file.read_text().strip()
        if not re.search(pattern, result.stderr):
            return fail(f"stderr does not match /{pattern}/", result.stdout,
                        result.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
