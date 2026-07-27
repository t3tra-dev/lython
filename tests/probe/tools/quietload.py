#!/usr/bin/env python3
"""Wait for the machine to go quiet, and report the load a measurement ran at.

rfc/lane-conversion-playbook.md section 5 says "never measure while a build is
running", and records a 12-cell grid taken at load 44 reporting 11/12 where the
same grid on a quiet machine reported 12/12. The rule was written as advice, and
advice is not enough here: this machine is shared with sibling agents and with the
coordinator, so whether a build starts halfway through a grid is not something the
measuring track controls. It happened again on 2026-07-27 -- load 7 to 81
mid-battery, and the concurrent ctest reported a golden as Failed against 459/459
on a quiet machine.

Both halves of this module exist because of that:

  wait_for_quiet()  blocks until the 1-minute average is under a limit, so the
                    gate is enforced by the harness rather than hoped for
  load_note()       a one-line string to print BESIDE each result, so a future
                    reader can tell a quiet number from a contended one instead
                    of having to trust that the gate held

The second is the more important one. A gate that silently failed leaves results
that look clean, which is the same shape as the defect it is meant to prevent.

CHAINING STAGES, which is where this gets subtle. If one stage waits for another,
wait on a COMPLETION SENTINEL the first stage writes, never on the absence of its
process. Two reasons:

  - `pgrep -f <toolname>` is not agent-scoped in a checkout with ~50 worktrees, so
    it matches siblings and can block forever on someone else's run.
  - The failure DIRECTION has to match what the wait gates. A wait that gates a
    load-ADDING stage (a build) must fail by never firing, because firing early
    adds load to somebody's measurement. Absence-of-process has the wrong
    direction: killing the first stage -- exactly what a load spike makes you do --
    satisfies it instantly and launches the build into the contended window.

Both were hit for real on 2026-07-27, by two different tracks, within an hour.

WHAT CONTENTION CAN AND CANNOT DO, so a contended run is not discarded reflexively.
For a pass/fail instrument, load can invent a failure (a timeout, a starved run)
but cannot invent a pass: a use-after-free does not stop being one because the
machine is busy, and libgmalloc unmaps deterministically. So an all-green result
taken under load is still usable, and it is the REDS that have to be re-taken.
Quantitative results have no such asymmetry -- peak RSS, bytes-per-iteration and
wall-clock move in both directions under memory pressure, so those must be taken
quiet or not quoted.

    python3 tests/probe/tools/quietload.py --limit 12      # block, then print
    python3 tests/probe/tools/quietload.py --report        # print, never block

Exit code is 0 once quiet, or 1 if the deadline passed while still loaded -- so a
caller that must not measure under load can stop rather than measure anyway.
"""

from __future__ import annotations

import argparse
import os
import sys
import time


def load1() -> float:
    """The 1-minute load average, or 0.0 where the platform has none.

    The gate deliberately reads the 1-minute figure: it answers "is anything busy
    RIGHT NOW", which is the question a start decision needs.

    Its known weakness, observed rather than theorised: a spike that has just ended
    leaves the 5-minute average high while the 1-minute one has already fallen, so
    the gate opens into the tail of someone else's run. That happened on
    2026-07-27 -- a grid started at 1m 5.05 with 5m still at 7.94 from a sibling's
    ctest, and it came out 12/12, so the tail did no harm that time.

    Why the gate is not raised to cover it: for a ten-minute instrument the tail is
    a small fraction of the run, and gating on the 5-minute average would block for
    minutes after the machine is genuinely free -- which pushes a measuring track
    toward skipping the gate. The tail is handled by DISCLOSURE instead: the
    recorded note carries all three averages, so "started at 1m 5.05 / 5m 7.94" is
    visible in the result and a reader can discount it. Prefer a gate that is
    always used plus an honest record over a stricter gate that gets bypassed.
    """
    try:
        return os.getloadavg()[0]
    except (AttributeError, OSError):
        return 0.0


def load_note() -> str:
    """One line naming all three averages, for printing beside a result.

    All three and not just the 1-minute one: a grid that starts quiet can still
    have been contended for most of its run, and the 5- and 15-minute figures are
    what show that after the fact.
    """
    try:
        one, five, fifteen = os.getloadavg()
    except (AttributeError, OSError):
        return "load unavailable on this platform"
    return f"load {one:.2f} / {five:.2f} / {fifteen:.2f} (1m / 5m / 15m)"


def wait_for_quiet(limit: float = 12.0, timeout: float = 3600.0,
                   poll: float = 30.0, log=None) -> bool:
    """Block until the 1-minute load is below `limit`. True if it got there.

    Polls rather than sleeping a fixed time because the wait is usually short and
    occasionally very long, and a fixed sleep would have to be sized for the long
    case on every run.
    """
    deadline = time.monotonic() + timeout
    waited = False
    while True:
        current = load1()
        if current < limit:
            if waited and log:
                log(f"machine quiet at {current:.2f}, proceeding")
            return True
        if not waited and log:
            log(f"load {current:.2f} >= {limit:.2f}, waiting for a quiet machine")
        waited = True
        if time.monotonic() >= deadline:
            if log:
                log(f"still at {current:.2f} after {timeout:.0f}s, giving up")
            return False
        time.sleep(poll)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=float, default=12.0)
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--report", action="store_true",
                        help="print the load and exit without waiting")
    args = parser.parse_args()
    if args.report:
        print(load_note())
        return 0
    quiet = wait_for_quiet(args.limit, args.timeout,
                           log=lambda m: print(m, flush=True))
    print(load_note())
    return 0 if quiet else 1


if __name__ == "__main__":
    sys.exit(main())
