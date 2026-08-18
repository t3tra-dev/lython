#!/usr/bin/env python3
"""Same source, two builds, different answers => the program reads memory it does
not own.

ASan's allocator does not hand a freed block back promptly, so a use-after-free
reads intact data there and reused garbage in the normal build. ASan itself stays
QUIET for these -- the runtime frees through its own paths and the read lands in a
block the sanitizer no longer tracks -- so the DIVERGENCE between the two builds
is the instrument, not the sanitizer's report. Run `asan_sweep.py` for the errors
ASan does catch; run this for the ones it does not.

    python3 tests/probe/tools/allocdiff.py \
        build/bin/lyc build-fuzz/bin/lyc tests/golden/cases 6

Baselines (2026-08-18): tests/golden/cases 440 programs, 0 disagree.
tests/probe 343 programs, 0 disagree. The positive control is the field-alias
use-after-free (`old = t.rows; t.rows = []; old[0] = 9`), which this flags and
`asan_sweep.py` does not: the normal build prints a wrong length or raises, the
ASan build prints CPython's answer. A sweep with no positive control cannot tell
"nothing is wrong" from "the instrument is not connected".

build-fuzz is the ASan tree from CLAUDE.md's fuzzing section; it needs no fuzzer
target for this, only `bin/lyc`."""
import concurrent.futures, os, pathlib, subprocess, sys

if len(sys.argv) < 4:
    sys.exit(__doc__)
plain, asan, corpus = sys.argv[1], sys.argv[2], pathlib.Path(sys.argv[3])
jobs = int(sys.argv[4]) if len(sys.argv) > 4 else 6
aenv = dict(os.environ)
aenv["ASAN_OPTIONS"] = ("detect_leaks=0:detect_container_overflow=0:"
                        "allow_user_poisoning=0:abort_on_error=0")
files = sorted(p for p in corpus.glob("*.py"))

def one(binary, path, env):
    try:
        r = subprocess.run([binary, "jit", str(path)], capture_output=True,
                           text=True, timeout=300, stdin=subprocess.DEVNULL,
                           env=env, cwd=str(path.parent))
        return r.returncode, r.stdout
    except subprocess.TimeoutExpired:
        return -99, "<timeout>"

def run(path):
    prc, pout = one(plain, path, os.environ)
    arc, aout = one(asan, path, aenv)
    if prc == arc and pout == aout:
        return None
    return (path.name, prc, arc, pout[:70].replace("\n", "|"),
            aout[:70].replace("\n", "|"))

rows = []
with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
    for row in pool.map(run, files):
        if row:
            rows.append(row)
print(f"{len(files)} programs, {len(rows)} disagree between the two builds")
for name, prc, arc, pout, aout in rows:
    print(f"  {name}\n      plain rc={prc} [{pout}]\n      asan  rc={arc} [{aout}]")
