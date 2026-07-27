#!/usr/bin/env python3
"""Census of release-interface collisions: which memref widths are claimed by
more than one contract's deallocator.

Reads the runtime manifests. A width claimed by two contracts means
findDeallocatorForValueGroup cannot tell them apart on shape alone -- the
configuration that hid the set/frozenset use-after-free.
"""
import collections
import pathlib
import re
import sys

root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".")
lines = []
for path in sorted((root / "src/lython/runtime/modules").glob("*.mlir")):
    lines += [l for l in path.read_text().split("\n")
              if "ly.runtime.deallocator" in l]

SIG = re.compile(r"\s*func\.func (?:private )?@(\w+)\((.*?)\) attributes")
seen, groups, multi, unparsed = set(), collections.defaultdict(set), {}, []
for line in lines:
    m = SIG.match(line)
    if not m:
        unparsed.append(line[:70])
        continue
    name, params = m.group(1), m.group(2)
    if name in seen:
        continue
    seen.add(name)
    types = re.findall(r": (memref<[^>]*>|i\d+|f\d+)", params)
    contract = re.search(r'ly\.runtime\.contract = "([^"]+)"', line)
    contract = contract.group(1) if contract else "<no contract attr>"
    if len(types) != 1:
        multi[name] = (len(types), contract)
        continue
    groups[types[0]].add(contract)

print(f"deallocator declarations : {len(lines)}")
print(f"distinct functions       : {len(seen)}")
print(f"single-input             : {sum(len(v) for v in groups.values())}")
print(f"multi-input              : {len(multi)}")
if unparsed:
    print(f"UNPARSED (would be silently dropped): {len(unparsed)}")
    for u in unparsed:
        print("   ", u)
print()

tied = {w: v for w, v in groups.items() if len(v) > 1}
print(f"widths claimed by MORE THAN ONE contract: {len(tied)} of {len(groups)}")
for w, cs in sorted(tied.items(), key=lambda kv: (-len(kv[1]), kv[0])):
    print(f"  {w:16s} {len(cs)}-way")
    for c in sorted(cs):
        print(f"       {c}")
print()
uniq = {w: next(iter(v)) for w, v in groups.items() if len(v) == 1}
print(f"widths unique to one contract: {len(uniq)}")
for w, c in sorted(uniq.items()):
    print(f"  {w:16s} {c}")
