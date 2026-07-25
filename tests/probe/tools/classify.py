#!/usr/bin/env python3
"""Classify every probe against CPython 3.14, then re-check the survivors.

Phase 1 runs each probe under CPython and under `lyc jit` and assigns one of the
classes the facts table uses. Phase 2 re-runs only the probes that came out
correct, under libgmalloc: a use-after-free that the normal allocator hides
looks exactly like a pass in phase 1, and the guard allocator costs roughly
eight times as much, which is why it is not spent on probes already known to be
wrong.

    python3 tests/probe/tools/classify.py ./build/bin/lyc tests/probe out.json -j 5

The JSON it writes is the input to annotate.py, which stamps each probe's own
header with its verdict.

    OK       stdout and exit status both match CPython
    OK/GM    passes plainly but dies or misbehaves under libgmalloc
    SILENT   both exit 0, stdout differs -- the class that matters most
    SILENT!  lyc completed where CPython raised, or the exception types differ
    LOUD     rejected at compile time with a diagnostic
    VERIFY   rejected, but by an MLIR verifier failure rather than a diagnostic,
             i.e. not a rejection at the earliest static boundary
    CRASH    died on a signal or aborted (a double free lands here)
    TIMEOUT  exceeded the budget
    CPYERR   CPython could not run it, so the probe itself is wrong
"""

import argparse
import json
import os
import pathlib
import re
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

CPY = "/opt/homebrew/Frameworks/Python.framework/Versions/3.14/bin/python3.14"
GM = dict(DYLD_INSERT_LIBRARIES="/usr/lib/libgmalloc.dylib",
          MALLOC_PROTECT_BEFORE="1", MALLOC_FILL_SPACE="1")

VERIFIER_RE = re.compile(
    r"does not dominate this use|failed to legalize|verification failed|"
    r"Verifier failed|LLVM Translation failed|operand #\d+ does not dominate")
ABORT_RE = re.compile(r"non-positive refcount|Assertion|GuardMalloc.*freed|"
                      r"double free|PrintStackTrace")
# An MLIR diagnostic is `loc(fused<...hundreds of chars...>[...]): error: msg`.
# The location has to come off BEFORE the note is truncated, or the truncation
# budget is spent on the location and the message is the part that gets cut.
MLIR_LOC = re.compile(r"^.*?\berror:\s*")


def diagnostic(stderr):
    """The first real diagnostic message in `stderr`, location stripped."""
    for line in stderr.splitlines():
        if "error" in line.lower():
            return MLIR_LOC.sub("", line, count=1).strip()[:300]
    tail = stderr.strip().splitlines()
    return tail[-1][:300] if tail else ""


def run(cmd, env_extra=None, timeout=300.0):
    env = dict(os.environ)
    if env_extra:
        env.update(env_extra)
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                           env=env)
        return r.returncode, r.stdout, r.stderr
    except subprocess.TimeoutExpired as e:
        def t(s):
            if s is None:
                return ""
            return s.decode(errors="replace") if isinstance(s, bytes) else s
        return "timeout", t(e.stdout), t(e.stderr)


def classify(lyc, case):
    case = pathlib.Path(case)
    cpy_rc, cpy_out, cpy_err = run([CPY, str(case)])
    if cpy_rc == "timeout":
        return case.name, dict(cls="CPYERR", note="cpython timeout")
    ly_rc, ly_out, ly_err = run([str(lyc), "jit", str(case)])
    rec = dict(cpy_rc=cpy_rc, cpy_out=cpy_out, cpy_err=cpy_err[-1500:],
               ly_rc=ly_rc, ly_out=ly_out, ly_err=ly_err[-3000:])

    if ly_rc == "timeout":
        rec["cls"] = "TIMEOUT"
        return case.name, rec
    if (isinstance(ly_rc, int) and ly_rc < 0) or ABORT_RE.search(ly_err):
        rec["cls"] = "CRASH"
        first = [l for l in (ly_out + "\n" + ly_err).splitlines()
                 if "refcount" in l or "malloc" in l]
        rec["note"] = ((f"signal {-ly_rc}; " if isinstance(ly_rc, int) and ly_rc < 0
                        else "") + (first[0][:120] if first else ""))
        # Whether the right answer appeared before the abort distinguishes a
        # wrong computation from a wrong release.
        rec["out_before_crash_matches"] = (ly_out == cpy_out)
        return case.name, rec

    if ly_rc != 0:
        if cpy_rc != 0 and "Traceback" in ly_err:
            cpy_ty = (cpy_err.strip().splitlines() or [""])[-1].split(":")[0].strip()
            ly_ty = (ly_err.strip().splitlines() or [""])[-1].split(":")[0].strip()
            same = cpy_ty and cpy_ty == ly_ty and ly_out == cpy_out
            rec["cls"] = "OK" if same else "SILENT!"
            rec["note"] = f"cpython raised {cpy_ty!r}, lyc raised {ly_ty!r}"
            return case.name, rec
        rec["cls"] = "VERIFY" if VERIFIER_RE.search(ly_err) else "LOUD"
        rec["note"] = diagnostic(ly_err)
        return case.name, rec

    if cpy_rc != 0:
        rec["cls"] = "SILENT!"
        rec["note"] = "cpython raised, lyc completed cleanly"
        return case.name, rec
    if ly_out != cpy_out:
        rec["cls"] = "SILENT"
        rec["note"] = f"cpython={cpy_out!r} lyc={ly_out!r}"
        return case.name, rec
    rec["cls"] = "OK"
    return case.name, rec


def guard_check(lyc, name, path, want):
    gm_rc, gm_out, gm_err = run([str(lyc), "jit", path], GM, timeout=1200.0)
    gm_err = "\n".join(l for l in gm_err.splitlines()
                       if not l.startswith("GuardMalloc["))
    bad = gm_rc != 0 or gm_out != want or bool(ABORT_RE.search(gm_err))
    return name, bad, gm_rc, gm_out, gm_err[-2000:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("lyc", type=pathlib.Path)
    ap.add_argument("probes", type=pathlib.Path)
    ap.add_argument("out", type=pathlib.Path)
    ap.add_argument("-j", "--jobs", type=int, default=5)
    args = ap.parse_args()
    lyc = args.lyc.resolve()

    cases = sorted(str(p) for p in args.probes.glob("*.py"))
    results = {}
    # Threads rather than processes: these are all subprocess waits, and a pool
    # of processes each holding a multi-hundred-megabyte lyc gets killed under
    # memory pressure by the leak probes.
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        for i, (name, rec) in enumerate(
                ex.map(lambda c: classify(lyc, c), cases), 1):
            results[name] = rec
            print(f"[1/{i}/{len(cases)}] {rec['cls']:8} {name} "
                  f"{rec.get('note', '')[:100]}", flush=True)

    # The leak probes run tens of thousands of iterations; guarded, they do not
    # finish, and they are measured by RSS instead (see leak.py).
    todo = [(n, str(args.probes / n), results[n]["cpy_out"])
            for n in results
            if results[n]["cls"] == "OK" and not n.endswith("_big.py")]
    print(f"\n== phase 2: {len(todo)} correct probes under libgmalloc ==",
          flush=True)
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        for i, (name, bad, gm_rc, gm_out, gm_err) in enumerate(
                ex.map(lambda t: guard_check(lyc, *t), todo), 1):
            if not bad:
                print(f"[2/{i}/{len(todo)}] ok      {name}", flush=True)
                continue
            r = results[name]
            r["cls"] = "OK/GM"
            r.update(gm_rc=gm_rc, gm_out=gm_out, gm_err=gm_err)
            head = (gm_err.strip().splitlines() or [""])[0]
            r["note"] = (f"gmalloc rc={gm_rc} "
                         f"out_match={gm_out == r['cpy_out']} {head[:140]}")
            print(f"[2/{i}/{len(todo)}] OK/GM   {name} {r['note'][:120]}",
                  flush=True)

    args.out.write_text(json.dumps(results, indent=1))
    print("\n== summary ==")
    for k, v in Counter(r["cls"] for r in results.values()).most_common():
        print(f"  {k:8} {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
