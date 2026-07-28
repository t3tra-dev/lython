#!/usr/bin/env python3
"""Census of release-interface collisions: which memref widths are claimed by
more than one contract's deallocator.

Reads the runtime manifests. A width claimed by two contracts means
findDeallocatorForValueGroup cannot tell them apart on shape alone -- the
configuration that hid the set/frozenset use-after-free.

It also GATES the width reservations in
`lowering/Passes/Runtime/ABI/HandleWidthRegistry.h`: a width booked for a
contract that has not converted yet must stay unclaimed, and this exits non-zero
when one is taken.

Why a gate and not another printed row: that file's ledger was maintained by
hand, and the census exists because it was wrong for as long as it was
maintained that way -- five of fourteen widths turned out to be shared while the
file recorded four contracts as converted-and-safe. A reservation is exactly the
claim a single branch cannot check by grepping its own tree (`complex` collided
with `bytes` twice, at 4 and then at 6), so it is the one that has to fail a
build rather than inform a reader.
"""
import collections
import pathlib
import re
import sys

# Widths booked in HandleWidthRegistry.h for a contract that has NOT converted
# yet.  Delete the entry in the same commit that flips the contract's
# `ly.runtime.shape` -- at that point the contract itself claims the width and
# the reservation is what the manifest says.
RESERVED = {
    "memref<15xi64>": "builtins.int",
}

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
print()

# Reservations.  Checked against BOTH claim shapes, because a width is not free
# unless neither holds: a single-input release interface of its own, and any
# position of a multi-input one (the latter is mechanism (B), which fails
# silently by selecting another contract's deallocator).
print(f"reserved widths: {len(RESERVED)}")
violations = []
multi_positions = collections.defaultdict(set)
for line in lines:
    m = SIG.match(line)
    if not m:
        continue
    params = m.group(2)
    types = re.findall(r": (memref<[^>]*>|i\d+|f\d+)", params)
    if len(types) == 1:
        continue
    contract = re.search(r'ly\.runtime\.contract = "([^"]+)"', line)
    contract = contract.group(1) if contract else "<no contract attr>"
    for t in types:
        multi_positions[t].add(contract)

for width, reservee in sorted(RESERVED.items()):
    holders = sorted(groups.get(width, set()) - {reservee})
    inside = sorted(multi_positions.get(width, set()) - {reservee})
    if holders:
        violations.append(
            f"{width} is reserved for {reservee} but is the single-input "
            f"release interface of: {', '.join(holders)}")
    if inside:
        violations.append(
            f"{width} is reserved for {reservee} but appears inside the "
            f"multi-input release interface of: {', '.join(inside)} "
            f"-- mechanism (B), selects the wrong deallocator silently")
    if not holders and not inside:
        print(f"  {width:16s} {reservee:24s} unclaimed, reservation intact")

if violations:
    print()
    print("RESERVATION VIOLATED:")
    for v in violations:
        print(f"  {v}")
    print()
    print("Either the reserving track has converted (drop the RESERVED entry "
          "here and in HandleWidthRegistry.h), or two tracks picked the same "
          "width and one must move BEFORE the merge.")
    sys.exit(1)
