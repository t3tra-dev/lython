#!/usr/bin/env python3
"""Count acquisitions and releases per contract INSIDE `@__main__`, from the
refcount-elision MLIR.

Why scoped to `@__main__`: the module also contains the whole imported runtime
library, whose counts are constants of lowering and swamp the program's own.
A release that mechanism (A) failed to insert is missing from the USER function,
so that is the only region where the question is decidable.

Why NOT report the direction of a total: family A/B shrink toward a double free
and family C grows toward a leak, so the same movement carries opposite
meanings.  This prints the ACQUIRE and RELEASE counts side by side per contract
and never sums them.

INPUT VERIFICATION: prints bytes consumed, whether the dump marker was found,
and the extracted region's line count; refuses if `@__main__` is absent or the
region is empty, rather than reporting zeros.

    python3 -u tests/probe/tools/releaseaudit.py <ir-dump.txt> [more.txt ...]
"""
import collections
import pathlib
import re
import sys

# Allocating boundaries per tied width, from the manifests: these are the
# functions whose owned result is a handle at the tied width.
ACQUIRE = {
    "builtins.float(w3)": r"@(LyFloat_FromF64|LyFloat_Add|LyFloat_Sub|"
                          r"LyFloat_Mul|LyFloat_TrueDiv|LyLong_Float)\b",
    "builtins.range(w5)": r"@(LyRange_New)\b",
    "range_iterator(w5)": r"@(LyRange_Iter|LyRangeIterator_Iter|"
                          r"__ly_range_iterator_alloc)\b",
    "builtins.dict(w8)": r"@(LyDict_FromLength|LyDict_Copy|__ly_dict_alloc)\b",
    "_io.StringIO(w8)": r"@(LyStringIO_New|__ly_membuf_new)\b",
    "builtins.int(w2)": r"@(LyLong_FromI64|LyLong_Add)\b",
    "builtins.str(w2)": r"@(LyUnicode_FromBytes|LyFloat_Str|LyLong_Str)\b",
}
RELEASE = {
    "builtins.float(w3)": r"@LyFloat_DecRef\b",
    "builtins.range(w5)": r"@LyRange_DecRef\b",
    "range_iterator(w5)": r"@LyRangeIterator_DecRef\b",
    "builtins.dict(w8)": r"@LyDict_DecRef\b",
    "_io.StringIO(w8)": r"@LyStringIO_DecRef\b",
    "builtins.int(w2)": r"@LyLong_DecRef\b",
    "builtins.str(w2)": r"@LyUnicode_DecRef\b",
}

if len(sys.argv) < 2:
    sys.exit(__doc__)

rows = []
for arg in sys.argv[1:]:
    p = pathlib.Path(arg)
    if not p.exists():
        sys.exit(f"REFUSING: no such dump {p}")
    text = p.read_text(errors="replace")
    marker = text.count("LYTHON_IR_DUMP")
    # Extract the @__main__ region: from its `func.func @__main__` line to the
    # next top-level `func.func` at the same indentation.
    lines = text.split("\n")
    start = next((i for i, l in enumerate(lines)
                  if re.search(r"func\.func @__main__\b", l)), None)
    print(f"{p.name}: bytes={len(text)} markers={marker} "
          f"main_at_line={start}")
    if start is None:
        print("   REFUSING this file: no @__main__ region found")
        continue
    indent = len(lines[start]) - len(lines[start].lstrip())
    end = len(lines)
    for i in range(start + 1, len(lines)):
        l = lines[i]
        if l.strip().startswith("func.func") and \
                (len(l) - len(l.lstrip())) <= indent:
            end = i
            break
    region = "\n".join(lines[start:end])
    print(f"   region lines={end - start} chars={len(region)}")
    if end - start < 2:
        print("   REFUSING this file: region is empty")
        continue
    acq = {k: len(re.findall(v, region)) for k, v in ACQUIRE.items()}
    rel = {k: len(re.findall(v, region)) for k, v in RELEASE.items()}
    rows.append((p.name, acq, rel, end - start))

if not rows:
    sys.exit("REFUSING: no usable region in any input")

print()
keys = list(ACQUIRE)
print(f"{'contract(width)':22s} " + " ".join(
    f"{n[:20]:>22s}" for n, _, _, _ in rows))
for k in keys:
    cells = []
    for _, acq, rel, _ in rows:
        cells.append(f"{acq[k]:>9d}A {rel[k]:>9d}R")
    if any(acq[k] or rel[k] for _, acq, rel, _ in rows):
        print(f"{k:22s} " + " ".join(f"{c:>22s}" for c in cells))
print()
print("A = acquiring calls in @__main__, R = releasing calls in @__main__.")
print("A>0 with R==0 is the signature mechanism (A) would produce: the group")
print("never formed, so no release was inserted.  A>0 with R>0 means the")
print("resource WAS discovered and released on that path.")
