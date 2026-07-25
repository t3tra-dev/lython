#!/usr/bin/env python3
"""Check that every repo filename a document cites actually exists.

A citation that no longer resolves is not a small documentation problem: to a
reader who cannot find the file, a claim backed by a missing reproducer is
indistinguishable from a fabricated one. That risk grows with edit count, so it
is worth a mechanical pass before handing a document to anyone -- the facts
table in rfc/object-ownership-probe.md grew through fourteen appended addenda
while probes were being added and renamed, and did carry a stale pair.

    python3 tests/probe/tools/checkrefs.py <document> [repo-root] [--allow NAME ...]

Brace groups are expanded, so `flow_ifone_w{3list,2float}_plainbind.py` is
checked as two names. Filenames are resolved against the repo root and against
the directories where probes, harnesses, goldens and runtime library modules
live, so a document may cite either a bare name or a path.

--allow lists names that are expected NOT to resolve. The legitimate case is a
document that records a rename: naming the old file is the point, and it must
not be reported forever. Anything not allow-listed is a finding.

Exit code is the number of unresolved citations.
"""

import argparse
import pathlib
import re
import sys

# Directories a bare filename may live in.
SEARCH = ("tests/probe", "tests/probe/tools", "tests/golden/cases",
          "tests/golden/errors", "tests/unit", "examples",
          "src/lython/runtime/lib", "src/lython/runtime/modules")

CITATION = re.compile(r"[\w./{},*-]*?[\w-]+\.(?:py|mlir|stdout|exitcode|stderr-re)")
BRACES = re.compile(r"\{([^}]*)\}")
# Golden cases get cited by bare name -- `cases/foo`, `errors/bar` -- because
# that is how the suite names them. Requiring an extension made every one of
# those invisible, and an invisible citation is worse than an unresolved one:
# the tool reports "all resolve" and the silence reads as a pass. Found by
# applying k-4a's observation about negative results from instruments whose
# domain was never checked, to this instrument.
GOLDEN = re.compile(r"\b(cases|errors)/([A-Za-z_0-9]+)\b")


def expand(name):
    """`a_w{x,y}_b.py` -> {`a_wx_b.py`, `a_wy_b.py`}."""
    m = BRACES.search(name)
    if not m:
        return {name}
    out = set()
    for alt in m.group(1).split(","):
        out |= expand(name[:m.start()] + alt.strip() + name[m.end():])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("document", type=pathlib.Path)
    ap.add_argument("root", nargs="?", type=pathlib.Path,
                    default=pathlib.Path(__file__).resolve().parents[3])
    ap.add_argument("--allow", nargs="*", default=[],
                    help="names expected not to resolve (e.g. a recorded rename)")
    args = ap.parse_args()
    root = args.root.resolve()
    allow = set(args.allow)

    text = args.document.read_text()
    cited = set()
    for raw in CITATION.findall(text):
        cited |= expand(raw)
    # A golden cited by bare name resolves to its .py under tests/golden/.
    for sub, name in GOLDEN.findall(text):
        cited.add(f"tests/golden/{sub}/{name}.py")

    dirs = [root / d for d in SEARCH]
    missing, resolved, globs, allowed = [], 0, [], []
    for name in sorted(cited):
        if "*" in name:
            globs.append(name)
            continue
        base = pathlib.Path(name).name
        if (root / name).exists() or any((d / base).exists() for d in dirs):
            resolved += 1
        elif base in allow or name in allow:
            allowed.append(name)
        else:
            missing.append(name)

    print(f"cited: {len(cited)}  resolved: {resolved}  "
          f"allow-listed: {len(allowed)}  globs skipped: {len(globs)}")
    for label, items in (("glob (not checked)", globs),
                         ("allow-listed", allowed)):
        for i in items:
            print(f"  {label}: {i}")
    if missing:
        print("\nUNRESOLVED:")
        for m in missing:
            print(f"  {m}")
    else:
        print("\nall cited filenames resolve.")
    return len(missing)


if __name__ == "__main__":
    sys.exit(main())
