#!/usr/bin/env python3
"""Does this compiler agree with CPython 3.14 on the programs we already have?

Every defect in this audit was found the same way: run a program under `lyc`,
run it under `python3.14`, look at the difference. That loop was entirely by
hand, which means it only ever covered the program in front of me -- and the
worst bucket, a program that RUNS and prints the wrong answer, is invisible
unless someone happens to run that exact program both ways.

This runs the loop over a whole corpus and sorts the outcomes into buckets:

    AGREE      same stdout, both exited 0
    WRONG-DECLARED  stdout differs, and the program says why (expect-wrong)
    WRONG      both ran to completion, stdout differs      <-- the bad one
    GAP        CPython ran it, lyc refused it
    EXTRA      lyc ran it, CPython refused it
    BOTH-FAIL  neither accepted it (refusal parity)
    TIMEOUT    one of them did not finish

WRONG is what the project's "never silently mis-execute" rule exists to make
impossible, so a WRONG is a bug report on its own. GAP is the work queue: it
is where unimplemented surface shows up, and it is expected to be non-empty.
BOTH-FAIL is not automatically fine either -- two refusals can disagree about
WHY -- but this tool does not compare diagnostics, only whether they refuse.

    python3 tests/probe/tools/differential.py ./build/bin/lyc tests/probe

Regression detection needs a baseline, because a corpus this size always has
standing GAPs and re-reading them every run is how a new WRONG goes unnoticed.
--baseline compares against a checked-in classification and reports only what
MOVED; --update-baseline rewrites it. A move toward AGREE is progress, a move
away from it is a regression, and the exit code says which happened:

    0  nothing moved, or everything that moved improved
    1  something regressed (an AGREE that stopped agreeing, or a new WRONG)
    2  the run itself could not be trusted (no interpreter, bad corpus)

Not every program belongs here. One that prints an address, a time or a
hash order cannot agree with anything, and one that reads stdin or argv is
not being run the same way twice. Mark those in the corpus itself with a
`# differential: skip <reason>` comment on any line -- keeping the reason next
to the program is what stops the skip list from outliving the reason for it.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

# The buckets, worst first: report order, and the order that decides which
# transition counts as a regression (a move DOWN this list is an improvement).
BUCKETS = ["WRONG", "WRONG-DECLARED", "TIMEOUT", "EXTRA", "GAP", "BOTH-FAIL",
           "AGREE", "SKIP"]
IMPROVEMENT_RANK = {name: index for index, name in enumerate(BUCKETS)}


@dataclass
class Outcome:
    name: str
    bucket: str
    detail: str = ""


def skip_reason(source: Path) -> str | None:
    return marker_reason(source, "skip")


def declared_wrong(source: Path) -> str | None:
    """Does the program itself say its divergence is the point?

    `# differential: expect-wrong <reason>` was already being written in the
    corpus before anything read it, so wb_const_set_literal_fold.py -- forty
    lines documenting a CPython codegen fold this compiler does not do -- was
    reported as a fresh WRONG and, being newer than the baseline, as a
    REGRESSION. A survey that calls a documented difference a regression is the
    failure mode these tools exist to avoid, so the marker is honoured: it is
    still a difference and still printed, but under its own heading, and only a
    CHANGE in it moves the exit code.
    """
    return marker_reason(source, "expect-wrong")


def marker_reason(source: Path, marker: str) -> str | None:
    needle = f"# differential: {marker}"
    for line in source.read_text(errors="replace").splitlines():
        found = line.find(needle)
        if found >= 0:
            return line[found + len(needle) :].strip() or "no reason given"
    return None


def run(argv: list[str], cwd: Path, timeout: float) -> tuple[int, str] | None:
    try:
        done = subprocess.run(
            argv,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            stdin=subprocess.DEVNULL,
        )
    except subprocess.TimeoutExpired:
        return None
    return done.returncode, done.stdout


def classify(lyc: Path, interpreter: str, source: Path, workdir: Path,
             timeout: float) -> Outcome:
    reason = skip_reason(source)
    if reason:
        return Outcome(source.name, "SKIP", reason)

    # Each program gets its own working directory: a case that writes a file
    # would otherwise race the same name under -j, and the two runs of the
    # SAME program must not see each other's leftovers either.
    reference_dir = workdir / (source.stem + ".ref")
    subject_dir = workdir / (source.stem + ".sub")
    reference_dir.mkdir(parents=True, exist_ok=True)
    subject_dir.mkdir(parents=True, exist_ok=True)

    reference = run([interpreter, str(source.resolve())], reference_dir, timeout)
    subject = run([str(lyc.resolve()), "jit", str(source.resolve())], subject_dir,
                  timeout)
    if reference is None or subject is None:
        which = "CPython" if reference is None else "lyc"
        return Outcome(source.name, "TIMEOUT", f"{which} exceeded {timeout:g}s")

    reference_code, reference_out = reference
    subject_code, subject_out = subject
    if reference_code != 0 and subject_code != 0:
        return Outcome(source.name, "BOTH-FAIL", f"rc {reference_code}/{subject_code}")
    if reference_code != 0:
        return Outcome(source.name, "EXTRA", f"CPython rc {reference_code}")
    if subject_code != 0:
        return Outcome(source.name, "GAP", f"lyc rc {subject_code}")
    if reference_out != subject_out:
        difference = first_difference(reference_out, subject_out)
        if (declared := declared_wrong(source)) is not None:
            return Outcome(source.name, "WRONG-DECLARED", f"{difference} [{declared}]")
        return Outcome(source.name, "WRONG", difference)
    return Outcome(source.name, "AGREE")


def first_difference(reference: str, subject: str) -> str:
    reference_lines = reference.splitlines()
    subject_lines = subject.splitlines()
    for index in range(max(len(reference_lines), len(subject_lines))):
        want = reference_lines[index] if index < len(reference_lines) else "<no line>"
        got = subject_lines[index] if index < len(subject_lines) else "<no line>"
        if want != got:
            return f"line {index + 1}: CPython {want!r} vs lyc {got!r}"
    return "outputs differ only in trailing newline"


def read_baseline(path: Path) -> dict[str, str]:
    baseline: dict[str, str] = {}
    if not path.exists():
        return baseline
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        name, _, bucket = line.partition(" ")
        baseline[name] = bucket.strip()
    return baseline


def write_baseline(path: Path, outcomes: list[Outcome], corpus: Path) -> None:
    lines = [
        "# Differential classification against CPython 3.14, by",
        "# tests/probe/tools/differential.py. Regenerate with --update-baseline.",
        f"# Corpus: {corpus}",
        "#",
        "# A name listed here as AGREE is a program this compiler runs exactly as",
        "# CPython does. Anything else is a standing difference, and the tool",
        "# reports only MOVEMENT against this file so a new one is visible.",
        "",
    ]
    for outcome in sorted(outcomes, key=lambda o: o.name):
        lines.append(f"{outcome.name} {outcome.bucket}")
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("lyc", type=Path, help="the lyc binary to test")
    parser.add_argument("corpus", type=Path, help="directory of .py programs")
    parser.add_argument("--interpreter", default="python3.14",
                        help="reference interpreter (default: python3.14)")
    parser.add_argument("--baseline", type=Path,
                        help="classification file to compare against")
    parser.add_argument("--update-baseline", action="store_true",
                        help="rewrite --baseline from this run")
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 4)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--workdir", type=Path,
                        help="scratch root (default: a temp dir next to the corpus)")
    parser.add_argument("names", nargs="*",
                        help="restrict to these case names (stem or filename)")
    args = parser.parse_args()

    if not args.lyc.exists():
        print(f"no such binary: {args.lyc}", file=sys.stderr)
        return 2
    # The reference interpreter is 3.14 on purpose: this project tracks that
    # version's behaviour, and a 3.12 would report its own changes as defects.
    probe = subprocess.run([args.interpreter, "-c", "import sys; print(sys.version_info[:2])"],
                           capture_output=True, text=True)
    if probe.returncode != 0:
        print(f"reference interpreter {args.interpreter!r} is not runnable",
              file=sys.stderr)
        return 2
    if probe.stdout.strip() != "(3, 14)":
        print(f"reference interpreter is {probe.stdout.strip()}, expected (3, 14)",
              file=sys.stderr)
        return 2

    sources = sorted(p for p in args.corpus.glob("*.py"))
    if args.names:
        wanted = {n.removesuffix(".py") for n in args.names}
        sources = [p for p in sources if p.stem in wanted]
    if not sources:
        print(f"no .py programs in {args.corpus}", file=sys.stderr)
        return 2

    workdir = args.workdir or (args.corpus.parent / ".differential-work")
    workdir.mkdir(parents=True, exist_ok=True)

    outcomes: list[Outcome] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [
            pool.submit(classify, args.lyc, args.interpreter, source, workdir,
                        args.timeout)
            for source in sources
        ]
        for future in concurrent.futures.as_completed(futures):
            outcomes.append(future.result())

    by_bucket: dict[str, list[Outcome]] = {name: [] for name in BUCKETS}
    for outcome in outcomes:
        by_bucket[outcome.bucket].append(outcome)

    for bucket in BUCKETS:
        entries = sorted(by_bucket[bucket], key=lambda o: o.name)
        if not entries:
            continue
        print(f"\n{bucket} ({len(entries)})")
        # AGREE and SKIP are counted, not listed: they are the majority and
        # printing them buries the buckets that need reading.
        if bucket in ("AGREE", "SKIP"):
            continue
        for outcome in entries:
            print(f"  {outcome.name:<52} {outcome.detail}")

    if args.baseline and args.update_baseline:
        write_baseline(args.baseline, outcomes, args.corpus)
        print(f"\nbaseline written: {args.baseline}")
        return 0

    if not args.baseline:
        print(f"\n{len(outcomes)} programs, {len(by_bucket['AGREE'])} agree")
        return 1 if by_bucket["WRONG"] else 0


    baseline = read_baseline(args.baseline)
    regressions: list[str] = []
    improvements: list[str] = []
    for outcome in sorted(outcomes, key=lambda o: o.name):
        was = baseline.get(outcome.name)
        if was is None:
            # A program the baseline has never seen is not a regression, but a
            # new one that is already WRONG is worth the same attention.
            if outcome.bucket == "WRONG":
                # WRONG-DECLARED is deliberately not here: the file it came
                # from is where the difference is explained.
                regressions.append(f"{outcome.name}: new, and {outcome.bucket}")
            continue
        if was == outcome.bucket:
            continue
        moved = f"{outcome.name}: {was} -> {outcome.bucket}"
        if IMPROVEMENT_RANK[outcome.bucket] > IMPROVEMENT_RANK.get(was, 0):
            improvements.append(moved)
        else:
            regressions.append(moved)

    missing = sorted(set(baseline) - {o.name for o in outcomes})
    if improvements:
        print(f"\nimproved ({len(improvements)})")
        for line in improvements:
            print(f"  {line}")
    if missing:
        print(f"\ngone from the corpus ({len(missing)})")
        for name in missing:
            print(f"  {name}")
    if regressions:
        print(f"\nREGRESSED ({len(regressions)})")
        for line in regressions:
            print(f"  {line}")
        return 1
    print("\nno regressions against the baseline")
    return 0


if __name__ == "__main__":
    sys.exit(main())
