# WHAT: `iter(s)` as a value over a set: only indexable sequences become an
#   iterator value; a for loop over the set works.
# KIND: 未対応 (unsupported)
# FOUND: 2026-10-20, porting unittest -- docs/unittest-port.md lists every ut_ probe.
# CLASSIFICATION @ lang-gaps fa5c535a: 3 loud 拒否 (診断)
#   iter() as a value requires indexable sequences (list/str/tuple/bytes); iterate non-indexable sources directly in a for loop, or convert with list(...) first
# CPython 3.14 expects: 5
def first(s: set[int]) -> int:
    it = iter(s)
    return next(it)


print(first({5}))
