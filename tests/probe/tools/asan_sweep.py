#!/usr/bin/env python3
"""Run every program in a corpus under the ASan-instrumented lyc (JIT) and report
AddressSanitizer errors: use-after-free, double-free, buffer overflow.

    python3 tests/probe/tools/asan_sweep.py build-fuzz/bin/lyc tests/golden/cases 6

The JIT runs the compiled program inside the compiler process, so an ASan build of
`lyc` instruments the PROGRAM's allocations as well as the compiler's -- which is
what makes this a memory-safety sweep over the corpus and not just over the
compiler.

Leak detection is OFF: tests/leak_gate.py and leak_sweep.py already measure leaks
per program, and ASan's leak output would bury the corruption reports this is for.

Baselines (2026-08-18): tests/golden/cases 440 programs, 1 not clean (`sys_exit`,
rc=5 by design). tests/probe 343 programs, 12 not clean, all of them rc=1
compile refusals and none of them an ASan error.

⛔ It is quiet on a use-after-free whose read lands in a block the runtime freed
through its own path: see allocdiff.py, which catches those by disagreement."""
import concurrent.futures, os, pathlib, subprocess, sys

if len(sys.argv) < 3:
    sys.exit(__doc__)
lyc, corpus = sys.argv[1], pathlib.Path(sys.argv[2])
jobs = int(sys.argv[3]) if len(sys.argv) > 3 else 6
env = dict(os.environ)
env["ASAN_OPTIONS"] = ("detect_leaks=0:detect_container_overflow=0:"
                       "allow_user_poisoning=0:abort_on_error=0:"
                       "halt_on_error=1:print_stacktrace=1")
files = sorted(p for p in corpus.glob("*.py"))

def run(path):
    try:
        # Resolved, because the run happens in the PROGRAM's directory (a case
        # that writes a file must not write into the repo root): a relative
        # corpus path then names nothing from there, and every program comes
        # back rc=1 -- a whole corpus reported "not clean" for a reason that has
        # nothing to do with memory.
        r = subprocess.run([lyc, "jit", str(path.resolve())], capture_output=True,
                           text=True, timeout=300, stdin=subprocess.DEVNULL,
                           env=env, cwd=str(path.parent))
    except subprocess.TimeoutExpired:
        return path.name, "timeout", ""
    blob = r.stdout + r.stderr
    for kind in ("heap-use-after-free", "double-free", "attempting double-free",
                 "heap-buffer-overflow", "stack-use-after-scope",
                 "SEGV on unknown address", "attempting free on address"):
        if kind in blob:
            head = [l for l in blob.splitlines() if "ERROR: AddressSanitizer" in l]
            return path.name, kind, (head[0] if head else "")[:160]
    return path.name, "ok" if r.returncode == 0 else f"rc={r.returncode}", ""

bad = []
with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
    for name, kind, detail in pool.map(run, files):
        if kind not in ("ok",):
            bad.append((name, kind, detail))
print(f"{len(files)} programs, {len(bad)} not clean")
for name, kind, detail in sorted(bad, key=lambda b: b[1]):
    print(f"  {name:52s} {kind} {detail}")
