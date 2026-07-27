#!/usr/bin/env python3
"""Run probes under several allocator regimes, not just one.

`flaky.py` answers "is the outcome stable under repetition". This answers a
different question that repetition cannot: "is the outcome stable under a
DIFFERENT heap layout". They are not the same, and the distinction decides
whether a use-after-free family that stopped failing is actually fixed or merely
no longer landing on a reused block.

The regimes probe different things, so a shape passing all of them is a much
stronger statement than a shape passing one many times:

  plain             the system allocator, which reuses freed blocks eagerly --
                    a use-after-free here usually reads back plausible data
  guard-after       libgmalloc, guard page AFTER the block: catches overruns and
                    unmaps the block on free, so any later access faults
  guard-before      libgmalloc, guard page BEFORE the block: shifts every
                    allocation's alignment and catches underruns instead, so it
                    is a different LAYOUT as well as a different check
  scribble          system allocator filling freed memory with 0x55: a
                    use-after-free READ returns garbage rather than the old
                    value, which catches the reads that survive unmapping
                    because they never happen after the free in wall-clock order
  prescribble       system allocator filling fresh memory with 0xAA: catches a
                    read of a slot that was never initialised, which is how a
                    dropped store presents

A shape that passes plain but fails any guarded or scribbled regime is a live
memory-safety defect whose visibility depends on allocator behaviour -- the class
this corpus exists to make deterministic.

    python3 tests/probe/tools/allocregimes.py ./build/bin/lyc tests/probe/alias_*.py
    python3 tests/probe/tools/allocregimes.py ./build/bin/lyc --regimes prescribble \
        tests/golden/cases/*.py

`--regimes` selects a subset, which matters because `prescribble` alone is worth
running over far more code than the full set is. k-4a's observation, from using
it on stage 4a: a dropped store into a heap slot reads back as whatever filled
the slot, so under the system allocator it surfaces as a plausible `0` and needs
a differential against CPython to notice, while under `prescribble` it surfaces
as garbage or a fault and announces itself. Both silent bugs that stage found had
printed `0`. So this is the only instrument here that catches a dropped store
WITHOUT an oracle -- the rest of this corpus needs CPython to say what the answer
should have been. That makes it the one regime cheap enough, and general enough,
to sweep a whole suite with periodically.

The expectation has TWO parts and this tool needed both. It read `.stdout` and
assumed status 0, so `sys_exit.py` -- which deliberately ends with 5, and says so
in an `.exitcode` sidecar -- came back `reject <-- FAILS` on a sweep where its
output matched exactly. Same domain hole as requiring a filename extension, found
the same way: by pointing this at inputs the workflow it was built for does not
produce. It now reads `.exitcode` when there is one.

The oracle is CPython by default, which bounds what can be checked: a spelling
CPython cannot run -- a Lython extension, a bare-annotated class constructed with
arguments -- comes back SILENT even when it is correct, which is a false positive
of exactly the shape this file's own docstring warns about. So a `.stdout`
sidecar beside the input is used instead when one exists, which is how the golden
suite already states its expectation. k-4a hit the false SILENT running this over
goldens; the row belongs in the instrument table either way, because the domain
is a property worth stating even now that it is wider.

VALIDATING THE SIDECAR PATH, permanently and on any tree:

    python3 tests/probe/tools/allocregimes.py <lyc> --regimes plain -n 1 \
        tests/probe/tools/fixtures/sentinel_wrong_expectation.py

That fixture prints one thing and its sidecar says another, so it must report
`SILENT(actual) [sidecar]` and exit non-zero. Unlike pointing this tool at a
known-broken probe, the fixture cannot be repaired: the mismatch is its purpose,
so it keeps validating the comparison after every real defect has been fixed.
k-4a's idea, and the reason it matters is one of theirs too -- evidence recorded
from a broken tree describes a binary that will not exist for long.

It validates the sidecar comparison and nothing else. The CPython-oracle path
must not get a deliberate mismatch, because there a mismatch IS a defect and a
planted one would read as a real finding.

THE CPYTHON PATH NO LONGER HAS A KNOWN-BROKEN INPUT (stage 4b). Every alias shape
in this corpus now passes all five regimes, `alias_read_mutate_nowriteback_dict`
included, so pointing this tool at `tests/probe/alias_*.py` exercises the regimes
but never the reporting. What is left, and it is worth being explicit that it is
less:

  - the sidecar fixture above shows a FAILURE can be printed and counted;
  - the regimes themselves are only exercised by inputs that pass them.

So a green sweep here says "these shapes survive five layouts", not "this tool
would notice if they did not". The second claim needs a tree where they do not --
the rebuild recipe is in `leak.py`, and `flaky.py --self-test` covers the
neighbouring question for the repetition tool.

Exit code is the number of probes failing in at least one selected regime.
"""

import argparse
import os
import pathlib
import re
import subprocess
import sys

CPY = "/opt/homebrew/Frameworks/Python.framework/Versions/3.14/bin/python3.14"

REGIMES = {
    "plain": {},
    "guard-after": dict(DYLD_INSERT_LIBRARIES="/usr/lib/libgmalloc.dylib",
                        MALLOC_FILL_SPACE="1"),
    "guard-before": dict(DYLD_INSERT_LIBRARIES="/usr/lib/libgmalloc.dylib",
                         MALLOC_PROTECT_BEFORE="1", MALLOC_FILL_SPACE="1"),
    "scribble": dict(MallocScribble="1"),
    "prescribble": dict(MallocPreScribble="1", MallocScribble="1"),
}

REFCOUNT = re.compile(r"Ly_(?:Inc|Dec)Ref observed non-positive refcount")


def run(lyc, case, env_extra, timeout=900.0):
    env = dict(os.environ)
    env.update(env_extra)
    # libgmalloc announces itself on stderr; drop that so it is not mistaken
    # for a diagnostic.
    try:
        r = subprocess.run([str(lyc), "jit", str(case)],
                           capture_output=True, stdin=subprocess.DEVNULL,
                           text=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        return "timeout"
    err = "\n".join(l for l in r.stderr.splitlines()
                    if not l.startswith("GuardMalloc["))
    want = run.want
    # A non-zero exit is not by itself a failure: the golden suite states an
    # expected status in an `.exitcode` sidecar, and `sys_exit.py` deliberately
    # ends with 5. Reading only `.stdout` reported it as `reject <-- FAILS` on a
    # sweep where its output matched exactly -- the same domain hole as requiring
    # a file extension, found the same way, by feeding this tool inputs the
    # workflow it was built for does not produce.
    if r.returncode == run.want_rc:
        return "ok" if r.stdout == want else f"SILENT({r.stdout.strip()})"
    if REFCOUNT.search(r.stdout + err):
        return "abort"
    if r.returncode < 0:
        return f"sig{-r.returncode}"
    return "reject"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("lyc", type=pathlib.Path)
    ap.add_argument("probes", nargs="+", type=pathlib.Path)
    ap.add_argument("-n", "--runs", type=int, default=3,
                    help="runs per regime (default 3)")
    ap.add_argument("--regimes", default=None,
                    help="comma-separated subset, e.g. prescribble; "
                         f"choices: {','.join(REGIMES)}")
    args = ap.parse_args()
    lyc = args.lyc.resolve()

    if args.regimes:
        names = [r.strip() for r in args.regimes.split(",")]
        unknown = [r for r in names if r not in REGIMES]
        if unknown:
            ap.error(f"unknown regime(s): {', '.join(unknown)}")
    else:
        names = list(REGIMES)
    width = max(len(p.name) for p in args.probes)
    print(f"{'probe':{width}}  " + "  ".join(f"{n:>12}" for n in names))
    print("-" * (width + 2 + 14 * len(names)))

    bad = 0
    for p in args.probes:
        # A `.stdout` sidecar is the suite's own statement of the expectation,
        # and it covers spellings CPython cannot run. Prefer it; fall back to
        # CPython for probes, which have no sidecar.
        sidecar = p.with_suffix(".stdout")
        if sidecar.exists():
            run.want, oracle = sidecar.read_text(), "sidecar"
        else:
            run.want = subprocess.run([CPY, str(p)],
                                      capture_output=True, stdin=subprocess.DEVNULL,
                                      text=True).stdout
            oracle = "cpython"
        # And the expected STATUS, which the suite states separately.
        status = p.with_suffix(".exitcode")
        run.want_rc = int(status.read_text().strip()) if status.exists() else 0
        cells, failed = [], False
        for name in names:
            seen = {run(lyc, p, REGIMES[name]) for _ in range(args.runs)}
            cell = "ok" if seen == {"ok"} else "/".join(sorted(seen))
            if seen != {"ok"}:
                failed = True
            cells.append(cell)
        bad += failed
        # Naming the oracle matters: a reader has to know whether a SILENT is a
        # disagreement with CPython or with a checked-in expectation.
        flag = f"   [{oracle}]" + ("   <-- FAILS" if failed else "")
        print(f"{p.name:{width}}  " + "  ".join(f"{c:>12}" for c in cells) + flag,
              flush=True)

    print(f"\n{bad}/{len(args.probes)} probes fail in at least one regime "
          f"({args.runs} runs per regime, {len(names)} regimes)")
    return bad


if __name__ == "__main__":
    sys.exit(main())
