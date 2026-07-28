#!/usr/bin/env python3
"""Acceptance check for the sequence-literal source-move repair (three defects).

The repair has to satisfy three conditions AT ONCE, and the reason this exists as
a script rather than as prose is that two candidate repairs each satisfied one
condition while silently breaking another:

  A  NO REGRESSION -- none of the 40 golden cases below may be newly refused.
     That list is the measured footprint of "skip slot-absorption retains in the
     affine walk", which looked like the fix for condition B.
  B  CONVERGENCE -- `s = "abc"` / `for k in range(3): t = (s,)` must COMPILE AND
     RUN. It is refused today with `ownership CFG exploration exceeded 20000
     states`: a slot retain inside a loop makes `state.retained`, part of the
     visited-state key, increase every iteration.
  C  NO SILENT MIS-EXECUTION -- `v = ys[0]` inside a nested loop must not produce
     exit 0 with a wrong value. It is REFUSED today, which is safe; the combined
     candidate repair turned that refusal into a silent wrong answer, which is
     the one direction this family may never move in.
  D  THE ACTUAL DEFECT -- the nested-loop over-release must be gone: correct
     value, every rep, no abort.

The designed-but-unattempted repair for B is a modelling change, not a predicate
flip: the container's release must DISCHARGE the slot retains it absorbed
(`aggregate(parent, path)` answered by `parent`), rather than the walk pretending
the retain never happened.

INPUT VERIFICATION: prints the binary's sha256, the population size for every
criterion, and how many programs it actually compiled/ran, before any verdict.
Refuses -- rather than reporting a pass -- if the golden corpus is missing, if
any baseline name is absent from it, or if it examined nothing. A false
all-clear is the failure mode that matters here, because what is being looked
for is the ABSENCE of regressions.

DOMAIN, and why criterion A does not simply count non-zero exits. `--emit-llvm`
is used as the compile oracle, and it is NOT total over the golden corpus:
`comprehension.py` fails it on SHIPPED with `cannot build executable: symbol
'main' already exists`, which is a link-step collision and says nothing about
ownership. The original sweep justified a single arm by pointing at `ctest -j8`
480/480 -- but ctest drives `run_case.py`, which uses `jit`, so that baseline was
taken through a DIFFERENT oracle and did not transfer. One of the 40 names is
therefore a false positive of that sweep; the measured footprint is 39.
So A classifies the diagnosis and counts only OWNERSHIP refusals, reporting
anything else as EXCLUDED (out of domain) instead of scoring it.

    python3 -u tests/probe/tools/seqlit_acceptance.py <path-to-lyc> [repo-root]
"""
import os
import pathlib
import subprocess
import sys

# The 40 golden cases newly refused by the slot-retain-skip experiment, measured
# 2026-07-28 against a tree where `ctest -j8` was 480/480 green, so every one of
# them compiles on shipped. 37 failed with `used after release`, 3 with
# `released or transferred more than once`.
BASELINE_REGRESSIONS = """
bytes_one_lane_interior class_object_field_ops class_user_exceptions
collections_counter_arith collections_counter_basics collections_counter_eq
collections_counter_setops collections_counter_update comp_tuple_target
comprehension container_constructors container_element_alias_lifetime
cross_container_box_fronted_fields cross_enum_generic_handler
cross_except_star_views cross_exception_field_box_slot
cross_float_range_contracts_fields cross_generator_lazy_chain
cross_itertools_constructors cross_nested_field_chain delete_item dict_build
dict_methods_complete generator_object_args iter_fusion_builtins
iter_lazy_values iterate itertools_chain_take itertools_combinatorics
itertools_pairwise_accumulate itertools_repeat_cycle_ziplongest
itertools_value_position loop_else loop_iterator_element_into_container_literal
optional_dict_get_narrowed set_comp sorted_iterables_and_keys_view str_compare
str_contains tuple_one_lane_interior
""".split()

COND_B = ('s = "abc"\ntlen = 0\nfor k in range(3):\n    t = (s,)\n'
          '    tlen = len(t)\nprint(len(s), tlen)\n')
COND_C = ('v = 0\nfor i in range(3, 4):\n    for j in range(2):\n'
          '        ys = [i]\n        v = ys[0]\nprint(v)\n')
COND_D = ('total = 0\nfor i in range(4):\n    for j in range(4):\n'
          '        ys = [i, j]\n        total += ys[0] + ys[1]\nprint(total)\n')

REPS = 5


def load():
    a, b, c = os.getloadavg()
    return "%.1f/%.1f/%.1f" % (a, b, c)


# ⛔ THIS LIST DECIDES WHETHER A REGRESSION IS SEEN AT ALL, so it is taken from
# the verifier's own message prefixes, not from the shapes one repair happened to
# produce. The first version enumerated three diagnoses by hand and MISSED
# `released owned resource ... is used by region terminator`; a candidate repair
# that refused 23 of these 40 cases was then printed as `ownership-refused 0 ;
# out-of-domain 23` and scored PASS. That is the exact failure this script exists
# to prevent -- a false all-clear when what is being looked for is an ABSENCE --
# and an out-of-domain count is not a safe place to put something unrecognised.
#
# Why prefixes and not whole messages: every affine-ownership diagnostic in
# verifier/runtime/AffineOwnership.cpp opens with one of these
# (`grep -oh '<< "[a-z][^"]*'` over that file enumerates them), and the suffix
# names the producer, which varies per program.
#
# Why the count is still reported separately rather than folded into the verdict:
# `--emit-llvm` genuinely is not total over this corpus -- a link-step collision
# (`symbol 'main' already exists`) says nothing about ownership -- so the
# category has to exist. It must just be narrow enough that nothing ownership-
# shaped can fall into it.
OWNERSHIP_DIAGNOSES = (
    "owned resource from",
    "conditionally owned resource from",
    "released owned resource from",
    "borrowed entry argument",
    "block argument",
    "ownership CFG exploration exceeded",
    "borrowed entry ownership CFG exploration exceeded",
    "ownership-consuming call only consumes part of",
    "generator lane",
    "generator resume",
)


def is_ownership_diagnosis(msg):
    return any(d in msg for d in OWNERSHIP_DIAGNOSES)


def run_source(lyc, tmp, name, src):
    """Run one inline program; return (verdict, rc, stdout) per rep."""
    path = tmp / (name + ".py")
    path.write_text(src)
    ref = subprocess.run([sys.executable, str(path)], capture_output=True,
                         text=True, timeout=180, stdin=subprocess.DEVNULL)
    if ref.returncode != 0:
        return None, None
    want = ref.stdout.strip()
    out = []
    for _ in range(REPS):
        r = subprocess.run([lyc, "jit", str(path)], capture_output=True,
                           text=True, timeout=900, stdin=subprocess.DEVNULL)
        if r.returncode == 0:
            out.append("." if r.stdout.strip() == want else "W")
        elif r.returncode == 1:
            out.append("R")
        else:
            out.append("X")
    return "".join(out), want


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    lyc = sys.argv[1]
    root = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else ".").resolve()
    if not os.path.isfile(lyc):
        print("REFUSE: no lyc at %s" % lyc)
        return 2
    sha = subprocess.run(["shasum", "-a", "256", lyc], capture_output=True,
                         text=True).stdout.split()[0]
    cases = root / "tests" / "golden" / "cases"
    if not cases.is_dir():
        print("REFUSE: no golden cases dir at %s" % cases)
        return 2

    print("lyc sha256   = %s" % sha)
    print("golden cases = %s" % cases)
    print("baseline set = %d names" % len(BASELINE_REGRESSIONS))
    print("reps         = %d per inline program" % REPS)
    print("load@start   = %s" % load())

    missing = [n for n in BASELINE_REGRESSIONS
               if not (cases / (n + ".py")).is_file()]
    if missing:
        print("REFUSE: %d baseline name(s) absent from the corpus -- the list is"
              " stale and a 'no regression' verdict would be meaningless:"
              % len(missing))
        for n in missing:
            print("    %s" % n)
        return 2
    print("baseline set verified present in the corpus: %d/%d"
          % (len(BASELINE_REGRESSIONS), len(BASELINE_REGRESSIONS)))
    print()

    # ---- A: no regression over the measured footprint ----------------------
    # Run from a scratch cwd: `lyc` drops `a.out` into the working directory, so
    # sweeping from the repo root pollutes the tree and, with a stale artifact
    # present, manufactures failures that look like refusals.
    scratch = pathlib.Path(os.environ.get("TMPDIR", "/tmp")) / "seqlit_acceptance_cwd"
    scratch.mkdir(parents=True, exist_ok=True)
    print("A  no-regression over the %d measured names" % len(BASELINE_REGRESSIONS))
    refused, excluded = [], []
    for n in BASELINE_REGRESSIONS:
        f = cases / (n + ".py")
        r = subprocess.run([lyc, "--emit-llvm", str(f)], capture_output=True,
                           text=True, timeout=900, stdin=subprocess.DEVNULL,
                           cwd=str(scratch))
        if r.returncode == 0:
            continue
        msg = ""
        for ln in r.stderr.splitlines():
            if "error:" in ln:
                msg = ln.split("error:", 1)[1].strip()[:90]
                break
        if is_ownership_diagnosis(msg):
            refused.append((n, msg))
        else:
            excluded.append((n, msg))
    print("   compiled %d/%d ; ownership-refused %d ; out-of-domain %d"
          % (len(BASELINE_REGRESSIONS) - len(refused) - len(excluded),
             len(BASELINE_REGRESSIONS), len(refused), len(excluded)))
    for n, m in refused:
        print("     REFUSED  %-45s %s" % (n, m))
    for n, m in excluded:
        print("     EXCLUDED %-45s %s" % (n, m))
        print("              (not an ownership diagnosis -- `--emit-llvm` is not"
              " total over this corpus; not scored)")
    a_ok = not refused
    print("   => %s" % ("PASS" if a_ok else "FAIL (%d refused)" % len(refused)))
    print()

    # ---- B, C, D -----------------------------------------------------------
    tmp = pathlib.Path(os.environ.get("TMPDIR", "/tmp")) / "seqlit_acceptance"
    tmp.mkdir(parents=True, exist_ok=True)
    results = {}
    for key, name, src, want_desc in (
            ("B", "cond_b_slot_retain_in_loop", COND_B, "runs, prints '3 1'"),
            ("C", "cond_c_nested_read_only", COND_C, "no SILENT wrong value"),
            ("D", "cond_d_nested_overrelease", COND_D, "runs, prints '48'")):
        marks, want = run_source(lyc, tmp, name, src)
        if marks is None:
            print("REFUSE: no CPython reference for %s" % key)
            return 2
        results[key] = marks
        print("%s  %-34s cpython=%-6s reps=%s" % (key, name, want, marks))

    b_ok = set(results["B"]) == {"."}
    # C's bar is deliberately weaker than "runs": a refusal is acceptable there,
    # a silent wrong answer is not. Conflating the two is what hid the
    # regression the first time.
    c_ok = "W" not in results["C"]
    d_ok = set(results["D"]) == {"."}
    print("   B => %s   (must run correctly)" % ("PASS" if b_ok else "FAIL"))
    print("   C => %s   (refusal OK, silent wrong NOT)"
          % ("PASS" if c_ok else "FAIL"))
    print("   D => %s   (the defect itself)" % ("PASS" if d_ok else "FAIL"))
    print()
    print("load@end = %s" % load())
    print()
    every = a_ok and b_ok and c_ok and d_ok
    print("ACCEPTANCE: %s" % ("PASS -- all four hold together"
                              if every else "FAIL"))
    if not every:
        print("  A=%s B=%s C=%s D=%s" % (a_ok, b_ok, c_ok, d_ok))
        print("  A repair satisfying only some of these has been measured twice"
              " and was worse than shipping nothing.")
    return 0 if every else 1


if __name__ == "__main__":
    sys.exit(main())
