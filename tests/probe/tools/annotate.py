#!/usr/bin/env python3
"""Stamp each probe's own header with its measured verdict.

A probe that changes class after a lowering change should explain itself without
a lookup, so the verdict lives in the file rather than only in a results table.

Rewriting is idempotent: only the lines between the CLASSIFICATION anchor and
the CPython anchor are replaced, so the `# probe:` description above them keeps
its own continuation lines across re-runs.

    python3 tests/probe/tools/classify.py ./build/bin/lyc tests/probe res.json
    python3 tests/probe/tools/leak.py     ./build/bin/lyc tests/probe leak.json
    python3 tests/probe/tools/annotate.py res.json tests/probe leak.json
"""

import argparse
import json
import pathlib
import re
import sys

LABEL = {
    "OK": "1 正しい",
    "SILENT": "2 silent 誤実行",
    "SILENT!": "2 silent 誤実行",
    "OK/GM": "2 silent use-after-free (libgmalloc で検出)",
    "LOUD": "3 loud 拒否 (診断)",
    "VERIFY": "3 loud 拒否 (MLIR verifier 失敗 = 最早境界での診断になっていない)",
    "CRASH": "4 クラッシュ / abort",
    "TIMEOUT": "4 タイムアウト",
    "CPYERR": "- (CPython が実行できず)",
}
STACK = re.compile(r"^#\s+#\d+ 0x")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", type=pathlib.Path)
    ap.add_argument("probes", type=pathlib.Path)
    ap.add_argument("leak", nargs="?", type=pathlib.Path, default=None)
    args = ap.parse_args()

    res = json.loads(args.results.read_text())
    leak = json.loads(args.leak.read_text()) if args.leak else {}

    n = 0
    for p in sorted(args.probes.glob("*.py")):
        rec = res.get(p.name)
        if not rec:
            continue
        note = (rec.get("note") or "").strip()
        if note.startswith("#0 0x"):
            note = ""  # a raw stack frame says nothing worth keeping

        leakline = ""
        if p.stem.startswith("leak_"):
            row = leak.get(p.stem[len("leak_"):].rsplit("_", 1)[0])
            if row and row.get("bytes_per_iter") is not None:
                b = row["bytes_per_iter"]
                verdict = ("リーク" if b > 500
                           else "リークなし (計測ノイズ ±130 B/回 の範囲)")
                leakline = f"# RSS: {b:.0f} バイト/回 → {verdict}"

        out, in_verdict = [], False
        for ln in p.read_text().splitlines():
            if ln.startswith("# CLASSIFICATION:"):
                in_verdict = True
                out.append(f"# CLASSIFICATION: {LABEL.get(rec['cls'], rec['cls'])}")
                if note and rec["cls"] != "OK":
                    out.append(f"#   {note[:220]}")
                continue
            if ln.startswith("# CPython 3.14 expects:"):
                in_verdict = False
                exp = (rec.get("cpy_out") or "").strip()
                out.append("# CPython 3.14 expects: "
                           + (exp.replace("\n", " / ") if exp else "(例外を送出)"))
                if leakline:
                    out.append(leakline)
                continue
            if in_verdict and (ln.startswith("#   ") or STACK.match(ln)):
                continue  # the previous run's detail line
            if ln.startswith("# RSS: "):
                continue
            out.append(ln)
        p.write_text("\n".join(out) + "\n")
        n += 1
    print(f"annotated {n} probes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
