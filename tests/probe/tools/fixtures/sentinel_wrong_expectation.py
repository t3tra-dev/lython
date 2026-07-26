# NOT A PROBE. This is a fixture that MUST fail, forever, on every tree.
#
# It exists to validate the sidecar-oracle path of allocregimes.py: the tool's
# negative result is "everything matched a checked-in expectation", and a
# negative result proves nothing unless the tool can be shown capable of
# emitting a positive one. So this program prints `actual` and its .stdout says
# `expected-different`. The mismatch IS the fixture -- it is not a defect, and
# there is nothing here to repair.
#
# Why this rather than pointing the tool at a known-broken probe: a real defect
# gets fixed. The domain test then silently stops testing anything, and the
# recorded evidence that the tool once worked describes a binary nobody has any
# more (the same tree-relativity that makes redcheck.py re-run its sentinel
# every time). This fixture is tree-independent and cannot be repaired away.
#
# It deliberately lives under tools/fixtures/ rather than in tests/probe/,
# because classify.py globs tests/probe/*.py and would count a permanent,
# intentional mismatch as a SILENT finding -- corrupting the very headline
# numbers this corpus reports.
#
# Scope, stated so nobody over-claims it: this validates ONLY the sidecar
# comparison. It cannot validate flaky.py's "no NOT STABLE", which needs genuine
# nondeterminism that no synthetic input can manufacture, and it must not be
# adapted to the CPython-oracle path, where a mismatch is a defect by definition
# and a deliberate one would be indistinguishable from a bug.
#
#     python3 tests/probe/tools/allocregimes.py <lyc> --regimes plain -n 1 \
#         tests/probe/tools/fixtures/sentinel_wrong_expectation.py
#
# Expected: SILENT(actual) [sidecar], flagged, exit non-zero. If it reports ok,
# the sidecar comparison is not comparing.

print("actual")
