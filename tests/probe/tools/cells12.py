#!/usr/bin/env python3
"""Fill the 12-cell acceptance grid for the object-ownership kernel redesign.

    {list, dict, io.StringIO} x {store, read} x {same frame, across a call}

The three contracts are chosen because they are physically different and
currently break in complementary ways: `list` is three lanes inlined into the
instance, `dict` is box-fronted with a five-lane payload, and `io.StringIO` is a
one-lane header-fronted native contract. A change that repairs only the store
direction, or only one lane width, cannot fill the grid.

    store  the field is REBOUND to a fresh value
    read   the field's value is read into a local, then mutated in a way that
           reallocates it (append / insert / write)

Each cell runs N times plainly and once under libgmalloc. The plain runs alone
cannot classify this space: several shapes are use-after-frees whose visible
face depends on allocator state, so the same program prints the right answer on
some runs and aborts on others. A cell counts as filled only when every plain
run matches CPython AND the guard-allocator run does too.

    python3 tests/probe/tools/cells12.py ./build/bin/lyc [-n RUNS] [--keep DIR]
    python3 tests/probe/tools/cells12.py ./build/bin/lyc --acquire inline
    python3 tests/probe/tools/cells12.py ./build/bin/lyc --realloc

Exit code is the number of unfilled cells, so a stage can gate on it.

`--acquire` picks the third axis, which the grid's twelve cells hold fixed: how
the frame gets the receiver. The default is a factory call, and `inline` builds
it in the frame. They are not interchangeable -- an inline-constructed instance
used to carry an owned-local marker that a lane re-root could republish while a
call-derived one could not, which is the asymmetry the redesign exists to remove
-- so a stage that fills the grid on one should be re-run on the other.

`--realloc` picks a FOURTH axis the grid also held fixed, and silently: whether
the read direction's mutation actually reallocates. A container is allocated with
capacity 64, so `[1]` plus one `append` reallocates nothing, and a cell filled
that way says nothing about whether the field observes a reallocation -- it only
says the field observes an in-place write. Measured at `kernel/4b`: with the
default init, `list read` filled both boundaries; with `--realloc` the same shape
crashed in libsystem_malloc before the interior-view repair. The default is off
because it is a different question, not a better one -- an in-place mutation is
worth measuring on its own -- but a stage that claims the read direction must
report both.
"""

import argparse
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
from collections import Counter

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import quietload  # noqa: E402  (same directory, resolved above)

CPY = "/opt/homebrew/Frameworks/Python.framework/Versions/3.14/bin/python3.14"
GM = dict(DYLD_INSERT_LIBRARIES="/usr/lib/libgmalloc.dylib",
          MALLOC_PROTECT_BEFORE="1", MALLOC_FILL_SPACE="1")
REFCOUNT = re.compile(r"Ly_(?:Inc|Dec)Ref observed non-positive refcount")

# Per contract: the field annotation, the initial value, a fresh replacement
# for the store direction, a reallocating mutation of a local alias for the
# read direction, and how to observe the field.
# `init_realloc` fills the container to its allocation capacity (64), so the
# read direction's single mutation has to grow it. io.StringIO needs no variant:
# its buffer grows on every write by construction.
CONTRACTS = {
    "list": dict(
        imports="",
        ann="list[int]",
        init="[1]",
        init_realloc="[" + ", ".join(str(i) for i in range(65)) + "]",
        fresh="[1, 2, 3]",
        mutate='ks.append(9)',
        alias_ann="list[int]",
        observe="len(n.f)",
    ),
    "dict": dict(
        imports="",
        ann="dict[str, int]",
        init='{"a": 1}',
        init_realloc="{" + ", ".join('"k%d": %d' % (i, i) for i in range(65)) + "}",
        fresh='{"a": 1, "b": 2, "c": 3}',
        mutate='ks["z"] = 9',
        alias_ann="dict[str, int]",
        observe="len(n.f)",
    ),
    "io.StringIO": dict(
        imports="import io\n\n\n",
        ann="io.StringIO",
        init="io.StringIO()",
        fresh="_fresh_sio()",
        mutate='ks.write("zzz")',
        alias_ann="io.StringIO",
        observe="len(n.f.getvalue())",
    ),
}

SIO_HELPER = """def _fresh_sio() -> io.StringIO:
    s = io.StringIO()
    s.write("abc")
    return s


"""


def program(contract, direction, boundary, acquire="call", realloc=False):
    c = CONTRACTS[contract]
    helper = SIO_HELPER if contract == "io.StringIO" else ""
    init = c["init_realloc"] if realloc and "init_realloc" in c else c["init"]
    head = f"""{c['imports']}class Node:
    def __init__(self, v: {c['ann']}) -> None:
        self.f: {c['ann']} = v


{helper}def make() -> Node:
    v: {c['ann']} = {init}
    return Node(v)


"""
    # How the frame GETS the receiver is its own axis, and the two halves of it
    # broke differently before the field store moved into a heap slot: an
    # inline-constructed instance carried an owned-local marker that a lane
    # re-root could republish, and a call-derived one had no such thing. A grid
    # filled on one says nothing about the other, so both spellings are here.
    acquisition = ("n = make()\n" if acquire == "call" else
                   f"v0: {c['ann']} = {init}\nn = Node(v0)\n")
    if direction == "store" and boundary == "same frame":
        body = acquisition + f"""fresh: {c['ann']} = {c['fresh']}
n.f = fresh
print({c['observe']})
"""
    elif direction == "store" and boundary == "across a call":
        body = f"""def rebind(n: Node) -> None:
    fresh: {c['ann']} = {c['fresh']}
    n.f = fresh


""" + acquisition + f"""rebind(n)
print({c['observe']})
"""
    elif direction == "read" and boundary == "same frame":
        body = acquisition + f"""ks: {c['alias_ann']} = n.f
{c['mutate']}
print({c['observe']})
"""
    else:
        body = f"""def touch(n: Node) -> None:
    ks: {c['alias_ann']} = n.f
    {c['mutate']}


""" + acquisition + f"""touch(n)
print({c['observe']})
"""
    return head + body


def run(cmd, env_extra=None, timeout=600.0):
    env = dict(os.environ)
    if env_extra:
        env.update(env_extra)
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                           env=env)
        return r.returncode, r.stdout, r.stderr
    except subprocess.TimeoutExpired:
        return "timeout", "", ""


def face(rc, out, err, want):
    if rc == "timeout":
        return "timeout"
    if rc == 0:
        return "ok" if out == want else f"SILENT({out.strip()})"
    if REFCOUNT.search(out + err):
        return "abort"
    if isinstance(rc, int) and rc < 0:
        return f"sig{-rc}"
    errs = [l for l in err.splitlines() if "error" in l.lower()]
    return "reject" if errs else f"exit{rc}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("lyc", type=pathlib.Path)
    ap.add_argument("-n", "--runs", type=int, default=6)
    ap.add_argument("--keep", type=pathlib.Path, default=None,
                    help="write the generated programs here instead of a temp dir")
    ap.add_argument("--acquire", choices=("call", "inline"), default="call",
                    help="how the frame gets the receiver: from a factory call "
                         "(default) or constructed in the frame")
    ap.add_argument("--realloc", action="store_true",
                    help="fill the container to capacity so the read "
                         "direction's mutation has to reallocate")
    ap.add_argument("--wait-load", type=float, default=None, metavar="N",
                    help="block until the 1-minute load average is under N "
                         "before measuring (see quietload.py: this grid has "
                         "twice reported a false unfilled cell under load)")
    args = ap.parse_args()
    lyc = args.lyc.resolve()

    if args.wait_load is not None:
        if not quietload.wait_for_quiet(args.wait_load,
                                        log=lambda m: print(m, flush=True)):
            print("refusing to measure under load", file=sys.stderr)
            return 99
    load_before = quietload.load_note()

    tmp = args.keep or pathlib.Path(tempfile.mkdtemp(prefix="cells12-"))
    tmp.mkdir(parents=True, exist_ok=True)

    rows = []
    unfilled = 0
    for contract in CONTRACTS:
        for direction in ("store", "read"):
            for boundary in ("same frame", "across a call"):
                slug = (f"{contract.replace('.', '_')}_{direction}_"
                        f"{boundary.replace(' ', '')}")
                p = tmp / f"{slug}.py"
                p.write_text(textwrap.dedent(program(contract, direction,
                                                     boundary, args.acquire,
                                                     args.realloc)))
                want = run([CPY, str(p)])[1]
                faces = Counter()
                for _ in range(args.runs):
                    faces[face(*run([str(lyc), "jit", str(p)]), want=want)] += 1
                gm = face(*run([str(lyc), "jit", str(p)], GM, timeout=1200.0),
                          want=want)
                filled = (list(faces) == ["ok"]) and gm == "ok"
                if not filled:
                    unfilled += 1
                rows.append((contract, direction, boundary, want.strip(),
                             dict(faces), gm, filled))

    w = max(len(r[0]) for r in rows)
    print(f"12-cell grid, {args.acquire}-acquired receiver"
          f"{', container filled to capacity' if args.realloc else ''}"
          f"  ({args.runs} plain runs + 1 libgmalloc run per cell)\n")
    print(f"{'contract':{w}}  {'dir':6} {'boundary':14} {'want':6} "
          f"{'plain':38} {'gmalloc':8} filled")
    print("-" * (w + 80))
    for contract, direction, boundary, want, faces, gm, filled in rows:
        print(f"{contract:{w}}  {direction:6} {boundary:14} {want:6} "
              f"{str(faces):38} {gm:8} {'YES' if filled else 'no'}")
    filled_n = sum(1 for r in rows if r[6])
    print(f"\nfilled {filled_n}/12")
    # Beside the result, not only before it: an unfilled cell here is read as a
    # regression in whatever change is being measured, and the first question is
    # whether the machine was busy. Printing both ends means a reader can answer
    # that from the output instead of trusting that the gate held for the whole
    # run -- the 5m/15m figures show contention the starting 1m figure missed.
    print(f"at start: {load_before}")
    print(f"at end:   {quietload.load_note()}")
    if args.keep is None:
        shutil.rmtree(tmp, ignore_errors=True)
    else:
        print(f"programs kept in {tmp}")
    return unfilled


if __name__ == "__main__":
    sys.exit(main())
