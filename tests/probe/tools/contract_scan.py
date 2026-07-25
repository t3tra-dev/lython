"""Narrow the candidates for declared-but-unimplemented contract methods.

For each `py.class` in `runtime/modules/*.mlir`, report the names in
`method_names` that have no `ly.runtime.method` / `ly.runtime.primitive` /
`ly.runtime.initializer` symbol under the same `ly.runtime.contract`.

This is a CANDIDATE list, not a defect list: a method reached through a C++
special path in the lowerer has no manifest symbol and shows up here as a false
positive. Confirm each candidate by differential execution against CPython --
contracts.py does that -- before reporting it. Skipping that step is the one
trap the side-defects track hit.

    python3 tests/probe/tools/contract_scan.py [repo-root]
"""

import pathlib
import re
import sys

ROOT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1
                    else pathlib.Path(__file__).resolve().parents[3])
MODULES = sorted((ROOT / "src/lython/runtime/modules").glob("*.mlir"))
if not MODULES:
    sys.exit(f"no manifests under {ROOT}/src/lython/runtime/modules")


def split_top(s):
    """Split a comma list at depth 0 of <>[](){}."""
    out, depth, cur = [], 0, ""
    for ch in s:
        if ch in "<[({":
            depth += 1
        elif ch in ">])}":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur.strip())
    return out


def parse_class_blocks(text):
    """Yield (class_name, attr_text) for each `py.class @N attributes { ... }`."""
    for m in re.finditer(r"py\.class\s+@(\w+)\s+attributes\s*\{", text):
        name = m.group(1)
        i = m.end() - 1
        depth = 0
        for j in range(i, len(text)):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    yield name, text[i + 1 : j]
                    break


def attr_list(attrs, key):
    m = re.search(re.escape(key) + r"\s*=\s*\[", attrs)
    if not m:
        return None
    i = m.end() - 1
    depth = 0
    for j in range(i, len(attrs)):
        if attrs[j] == "[":
            depth += 1
        elif attrs[j] == "]":
            depth -= 1
            if depth == 0:
                return split_top(attrs[i + 1 : j])
    return None


def attr_str(attrs, key):
    m = re.search(re.escape(key) + r'\s*=\s*"([^"]*)"', attrs)
    return m.group(1) if m else None


# --- gather implementations -------------------------------------------------
impl = {}  # contract -> {"method": set, "primitive": set, "initializer": set}
for path in MODULES:
    text = path.read_text()
    for m in re.finditer(r"func\.func\s+(?:private\s+)?@(\w+)\(", text):
        # attribute dict of this func (up to the opening body brace or EOL)
        tail = text[m.end() : text.find("\n", m.end()) + 1]
        c = re.search(r'ly\.runtime\.contract\s*=\s*"([^"]+)"', tail)
        if not c:
            continue
        d = impl.setdefault(
            c.group(1), {"method": set(), "primitive": set(), "initializer": set()}
        )
        for role in ("method", "primitive", "initializer"):
            r = re.search(r'ly\.runtime\.' + role + r'\s*=\s*"([^"]+)"', tail)
            if r:
                d[role].add(r.group(1))

# --- compare against typing contracts --------------------------------------
rows = []
for path in MODULES:
    text = path.read_text()
    for cls, attrs in parse_class_blocks(text):
        contract = attr_str(attrs, "ly.runtime.contract")
        if not contract:
            continue
        names = attr_list(attrs, "method_names") or []
        names = [n.strip('"') for n in names]
        kinds = [k.strip('"') for k in (attr_list(attrs, "method_kinds") or [])]
        have = impl.get(contract, {"method": set(), "primitive": set(), "initializer": set()})
        missing = []
        for idx, n in enumerate(names):
            if n in have["method"] or n in have["initializer"]:
                continue
            if n in have["primitive"] or (n + "_box") in have["primitive"]:
                continue
            kind = kinds[idx] if idx < len(kinds) else "?"
            missing.append(f"{n}[{kind}]")
        if missing:
            rows.append((path.name, cls, contract, sorted(set(missing))))

for fn, cls, contract, missing in rows:
    print(f"{fn}: @{cls} ({contract})")
    print("    " + ", ".join(missing))
