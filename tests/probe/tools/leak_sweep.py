#!/usr/bin/env python3
"""Which programs leak, over a whole corpus?

The leak gate is a REGRESSION gate: fourteen cases measured at net zero, and it
fails if that changes. It cannot find a leak in the other ~360 goldens, and
three entries in rfc/test-suite-debt.md are figures with no program attached --
"bounded, 2.8 KB", "bounded, 62 B", "9 roots / 22096 B" -- so there is nothing
to re-measure and nothing to fix. That is what this closes: it runs the gate's
own measurement over every program in a directory and prints the ones that are
not net zero, so a figure gets a reproducer.

    python3 tests/probe/tools/leak_sweep.py ./build/bin/lyc tests/golden/cases

Each case costs one sanitized AOT build plus one run, so this is minutes, not
seconds; --jobs spends cores on it. Cases the compiler refuses, or that expect
a non-zero exit, are reported as SKIP rather than counted -- a refusal carries
no leak information, and reading one as clean is how a survey lies.

⛔ Not a ctest stage, and not on a timer. A red test is not something to commit
(tests/CMakeLists.txt), and a survey over a corpus with known leakers in it is
red by construction. This is the tool that finds them; the gate is where a
repaired one gets locked in.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import pathlib
import re
import subprocess
import sys

NET = re.compile(r"net (\d+) allocs / (\d+) B")


def measure(runner: pathlib.Path, lyc: pathlib.Path, source: pathlib.Path,
            timeout: float) -> tuple[str, int, int, str]:
    try:
        done = subprocess.run(
            [sys.executable, str(runner), str(lyc), str(source)],
            capture_output=True, text=True, timeout=timeout,
            stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        return source.name, -1, -1, "timeout"
    text = done.stdout + done.stderr
    found = NET.search(text)
    if not found:
        # No measurement: a refused compile, an expected non-zero exit, or the
        # sanitizer going silent. None of those is a clean result.
        reason = "refused" if "emit error" in text or "lowering pipeline" in text \
            else ("could not measure" if done.returncode == 2 else "no summary")
        return source.name, -1, -1, reason
    return source.name, int(found.group(1)), int(found.group(2)), ""


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("lyc", type=pathlib.Path)
    parser.add_argument("corpus", type=pathlib.Path)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("names", nargs="*",
                        help="restrict to these case stems")
    args = parser.parse_args()

    runner = pathlib.Path(__file__).resolve().parents[2] / "leak_gate.py"
    if not runner.exists():
        print(f"leak gate runner not found: {runner}", file=sys.stderr)
        return 2
    sources = sorted(args.corpus.glob("*.py"))
    if args.names:
        wanted = {n.removesuffix(".py") for n in args.names}
        sources = [p for p in sources if p.stem in wanted]
    if not sources:
        print(f"no .py programs in {args.corpus}", file=sys.stderr)
        return 2

    leaking: list[tuple[str, int, int]] = []
    skipped: list[tuple[str, str]] = []
    clean = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(measure, runner, args.lyc, source, args.timeout)
                   for source in sources]
        for done in concurrent.futures.as_completed(futures):
            name, allocs, size, reason = done.result()
            if reason:
                skipped.append((name, reason))
            elif allocs == 0 and size == 0:
                clean += 1
            else:
                leaking.append((name, allocs, size))
                print(f"LEAK {name:<52} {allocs} allocs / {size} B", flush=True)

    print(f"\n{len(sources)} programs: {clean} clean, {len(leaking)} leaking, "
          f"{len(skipped)} not measured")
    if leaking:
        print("\nleaking, largest first")
        for name, allocs, size in sorted(leaking, key=lambda r: -r[2]):
            print(f"  {name:<52} {allocs} allocs / {size} B")
    if skipped:
        print("\nnot measured")
        for name, reason in sorted(skipped):
            print(f"  {name:<52} {reason}")
    return 1 if leaking else 0


if __name__ == "__main__":
    sys.exit(main())
