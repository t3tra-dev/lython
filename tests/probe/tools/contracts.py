"""Differential contract audit: run each container/str/bytes contract method
through lyc and through CPython 3.14 and report every disagreement.

A manifest declares a contract's `method_names`, but a declared name with no
implementation behind it is only visible when the call is actually made -- a
static scan cannot separate a missing implementation from one reached through a
C++ special path (see contract_scan.py, which narrows the candidates but cannot
decide them). So every case here is a whole program whose output CPython
defines, and disagreement in either stdout or exit status is the finding.

Written by the side-defects track, which found six declared-but-unimplemented
methods (list.pop, list.insert, tuple.__add__/__mul__/count/index) in the first
seven contracts it covered. Seventy-two of the seventy-nine contracts that
declare method_names are still unaudited, roughly 336 methods; the largest are
int (38), float (30), _asyncio.Future (14) and Task (12), the _io classes, and
object (11). Extend CASES rather than starting over.

    python3 tests/probe/tools/contracts.py ./build/bin/lyc [name-substring]

Exit code is the number of disagreements. Gate on "no more than the recorded
baseline", not on zero: the known unimplemented methods disagree by design until
they are implemented, so zero is only the right target once the audit is
finished. Unlike contract_scan.py this count IS reachable, because every case
here is decided by execution rather than by symbol lookup.

Baseline at c3de5e7: **10 disagreements of 215 cases** -- list.pop in three
spellings, list.insert in three, and tuple.__add__/__mul__/count/index. That is
the same six methods the side-defects track reported, arrived at on a different
tree, which is what makes it a cross-check rather than a repeat.
"""

import argparse
import pathlib
import subprocess
import sys
import tempfile

CPY = "/opt/homebrew/Frameworks/Python.framework/Versions/3.14/bin/python3.14"

CASES = {}


def case(name, src):
    CASES[name] = src


# ---------------- list ----------------
L = 'xs: list[int] = [3, 1, 2, 1]\n'
case("list.append", L + "xs.append(9)\nprint(xs)")
case("list.extend", L + "ys: list[int] = [7, 8]\nxs.extend(ys)\nprint(xs)")
case("list.pop", L + "v: int = xs.pop()\nprint(v)\nprint(xs)")
case("list.pop(i)", L + "v: int = xs.pop(0)\nprint(v)\nprint(xs)")
case("list.pop(-2)", L + "v: int = xs.pop(-2)\nprint(v)\nprint(xs)")
case("list.insert", L + "xs.insert(1, 99)\nprint(xs)")
case("list.insert(-1)", L + "xs.insert(-1, 99)\nprint(xs)")
case("list.insert(big)", L + "xs.insert(100, 99)\nprint(xs)")
case("list.remove", L + "xs.remove(1)\nprint(xs)")
case("list.clear", L + "xs.clear()\nprint(xs)")
case("list.__len__", L + "print(len(xs))")
case("list.__iter__", L + "t: int = 0\nfor x in xs:\n    t = t + x\nprint(t)")
case("list.__getitem__", L + "print(xs[2])")
case("list.__getslice__", L + "print(xs[1:3])")
case("list.__setslice__", L + "ys: list[int] = [0, 0]\nxs[1:3] = ys\nprint(xs)")
case("list.__delslice__", L + "del xs[1:3]\nprint(xs)")
case("list.__setitem__", L + "xs[0] = 42\nprint(xs)")
case("list.__delitem__", L + "del xs[0]\nprint(xs)")
case("list.__contains__", L + "print(2 in xs)")
case("list.__repr__", L + "print(repr(xs))")
case("list.sort", L + "xs.sort()\nprint(xs)")
case("list.reverse", L + "xs.reverse()\nprint(xs)")
case("list.copy", L + "ys: list[int] = xs.copy()\nprint(ys)")
case("list.count", L + "print(xs.count(1))")
case("list.index", L + "print(xs.index(2))")
case("list.__add__", L + "ys: list[int] = [5]\nprint(xs + ys)")
case("list.__mul__", "xs: list[int] = [1, 2]\nprint(xs * 2)")
case("list.__eq__", L + "ys: list[int] = [3, 1, 2, 1]\nprint(xs == ys)")
case("list.__ne__", L + "ys: list[int] = [3]\nprint(xs != ys)")
case("list.__lt__", L + "ys: list[int] = [4]\nprint(xs < ys)")
case("list.__le__", L + "ys: list[int] = [4]\nprint(xs <= ys)")
case("list.__gt__", L + "ys: list[int] = [4]\nprint(xs > ys)")
case("list.__ge__", L + "ys: list[int] = [4]\nprint(xs >= ys)")

# ---------------- dict ----------------
D = 'd: dict[str, int] = {"a": 1, "b": 2}\n'
case("dict.__len__", D + "print(len(d))")
case("dict.__iter__", D + "n: int = 0\nfor k in d:\n    n = n + 1\nprint(n)")
case("dict.__getitem__", D + 'print(d["a"])')
case("dict.get", D + 'print(d.get("a", 0))')
case("dict.__setitem__", D + 'd["c"] = 3\nprint(len(d))')
case("dict.__delitem__", D + 'del d["a"]\nprint(len(d))')
case("dict.__contains__", D + 'print("a" in d)')
case("dict.keys", D + "n: int = 0\nfor k in d.keys():\n    n = n + 1\nprint(n)")
case("dict.values", D + "t: int = 0\nfor v in d.values():\n    t = t + v\nprint(t)")
case("dict.items", D + "t: int = 0\nfor k, v in d.items():\n    t = t + v\nprint(t)")
case("dict.__repr__", D + "print(repr(d))")
case("dict.clear", D + "d.clear()\nprint(len(d))")
case("dict.copy", D + "e: dict[str, int] = d.copy()\nprint(len(e))")
case("dict.update", D + 'e: dict[str, int] = {"c": 3}\nd.update(e)\nprint(len(d))')
case("dict.__or__", D + 'e: dict[str, int] = {"c": 3}\nprint(len(d | e))')
case("dict.__eq__", D + 'e: dict[str, int] = {"a": 1, "b": 2}\nprint(d == e)')
case("dict.__ne__", D + 'e: dict[str, int] = {"a": 1}\nprint(d != e)')
case("dict.pop", D + 'print(d.pop("a"))')
case("dict.pop(default)", D + 'print(d.pop("z", 0))')
case("dict.popitem", D + "k, v = d.popitem()\nprint(v)")
case("dict.setdefault", D + 'print(d.setdefault("c", 3))')

# ---------------- set ----------------
S = "s: set[int] = {1, 2, 3}\n"
T = "t: set[int] = {3, 4}\n"
case("set.add", S + "s.add(9)\nprint(len(s))")
case("set.__len__", S + "print(len(s))")
case("set.__iter__", S + "n: int = 0\nfor x in s:\n    n = n + 1\nprint(n)")
case("set.__contains__", S + "print(2 in s)")
case("set.discard", S + "s.discard(2)\nprint(len(s))")
case("set.remove", S + "s.remove(2)\nprint(len(s))")
case("set.clear", S + "s.clear()\nprint(len(s))")
case("set.copy", S + "u: set[int] = s.copy()\nprint(len(u))")
case("set.update", S + T + "s.update(t)\nprint(len(s))")
case("set.intersection_update", S + T + "s.intersection_update(t)\nprint(len(s))")
case("set.difference_update", S + T + "s.difference_update(t)\nprint(len(s))")
case("set.symmetric_difference_update", S + T + "s.symmetric_difference_update(t)\nprint(len(s))")
case("set.union", S + T + "print(len(s.union(t)))")
case("set.intersection", S + T + "print(len(s.intersection(t)))")
case("set.difference", S + T + "print(len(s.difference(t)))")
case("set.symmetric_difference", S + T + "print(len(s.symmetric_difference(t)))")
case("set.issubset", S + T + "print(s.issubset(t))")
case("set.issuperset", S + T + "print(s.issuperset(t))")
case("set.isdisjoint", S + T + "print(s.isdisjoint(t))")
case("set.__eq__", S + "u: set[int] = {1, 2, 3}\nprint(s == u)")
case("set.__ne__", S + T + "print(s != t)")
case("set.__or__", S + T + "print(len(s | t))")
case("set.__and__", S + T + "print(len(s & t))")
case("set.__sub__", S + T + "print(len(s - t))")
case("set.__xor__", S + T + "print(len(s ^ t))")
case("set.__le__", S + T + "print(s <= t)")
case("set.__lt__", S + T + "print(s < t)")
case("set.__ge__", S + T + "print(s >= t)")
case("set.__gt__", S + T + "print(s > t)")

# ---------------- frozenset ----------------
F = "f: frozenset[int] = frozenset([1, 2, 3])\n"
G = "g: frozenset[int] = frozenset([3, 4])\n"
case("frozenset.__len__", F + "print(len(f))")
case("frozenset.__iter__", F + "n: int = 0\nfor x in f:\n    n = n + 1\nprint(n)")
case("frozenset.__contains__", F + "print(2 in f)")
case("frozenset.__hash__", F + "print(hash(f) == hash(f))")
case("frozenset.__eq__", F + "h: frozenset[int] = frozenset([1, 2, 3])\nprint(f == h)")
case("frozenset.__ne__", F + G + "print(f != g)")
case("frozenset.union", F + G + "print(len(f.union(g)))")
case("frozenset.intersection", F + G + "print(len(f.intersection(g)))")
case("frozenset.difference", F + G + "print(len(f.difference(g)))")
case("frozenset.symmetric_difference", F + G + "print(len(f.symmetric_difference(g)))")
case("frozenset.issubset", F + G + "print(f.issubset(g))")
case("frozenset.issuperset", F + G + "print(f.issuperset(g))")
case("frozenset.isdisjoint", F + G + "print(f.isdisjoint(g))")
case("frozenset.__or__", F + G + "print(len(f | g))")
case("frozenset.__and__", F + G + "print(len(f & g))")
case("frozenset.__sub__", F + G + "print(len(f - g))")
case("frozenset.__xor__", F + G + "print(len(f ^ g))")
case("frozenset.__le__", F + G + "print(f <= g)")
case("frozenset.__lt__", F + G + "print(f < g)")
case("frozenset.__ge__", F + G + "print(f >= g)")
case("frozenset.__gt__", F + G + "print(f > g)")

# ---------------- tuple ----------------
P = "p: tuple[int, int, int] = (3, 1, 2)\n"
case("tuple.__len__", P + "print(len(p))")
case("tuple.__contains__", P + "print(1 in p)")
case("tuple.__getitem__", P + "print(p[1])")
case("tuple.__getslice__", P + "print(p[0:2])")
case("tuple.__iter__", P + "t: int = 0\nfor x in p:\n    t = t + x\nprint(t)")
case("tuple.__add__", P + "q: tuple[int] = (9,)\nprint(p + q)")
case("tuple.__mul__", "p: tuple[int, int] = (1, 2)\nprint(p * 2)")
case("tuple.count", P + "print(p.count(1))")
case("tuple.index", P + "print(p.index(2))")
case("tuple.__repr__", P + "print(repr(p))")
case("tuple.__hash__", P + "print(hash(p) == hash(p))")
case("tuple.__eq__", P + "q: tuple[int, int, int] = (3, 1, 2)\nprint(p == q)")
case("tuple.__ne__", P + "q: tuple[int, int, int] = (3, 1, 9)\nprint(p != q)")
case("tuple.__lt__", P + "q: tuple[int, int, int] = (4, 1, 2)\nprint(p < q)")
case("tuple.__le__", P + "q: tuple[int, int, int] = (4, 1, 2)\nprint(p <= q)")
case("tuple.__gt__", P + "q: tuple[int, int, int] = (4, 1, 2)\nprint(p > q)")
case("tuple.__ge__", P + "q: tuple[int, int, int] = (4, 1, 2)\nprint(p >= q)")

# ---------------- str ----------------
A = 's: str = "Hello World"\n'
case("str.__len__", A + "print(len(s))")
case("str.__iter__", A + "n: int = 0\nfor c in s:\n    n = n + 1\nprint(n)")
case("str.__getitem__", A + "print(s[1])")
case("str.__getslice__", A + "print(s[1:4])")
case("str.__add__", A + 'print(s + "!")')
case("str.__contains__", A + 'print("World" in s)')
case("str.__mul__", 's: str = "ab"\nprint(s * 3)')
case("str.join", 'xs: list[str] = ["a", "b"]\nprint(",".join(xs))')
case("str.startswith", A + 'print(s.startswith("He"))')
case("str.startswith(s,e)", A + 'print(s.startswith("llo", 2, 5))')
case("str.endswith", A + 'print(s.endswith("ld"))')
case("str.endswith(s,e)", A + 'print(s.endswith("lo", 0, 5))')
case("str.__repr__", A + "print(repr(s))")
case("str.__str__", A + "print(str(s))")
case("str.encode", A + "print(len(s.encode()))")
case("str.upper", A + "print(s.upper())")
case("str.lower", A + "print(s.lower())")
case("str.casefold", A + "print(s.casefold())")
case("str.title", A + "print(s.title())")
case("str.capitalize", A + "print(s.capitalize())")
case("str.swapcase", A + "print(s.swapcase())")
case("str.isalpha", A + "print(s.isalpha())")
case("str.isspace", A + "print(s.isspace())")
case("str.isdecimal", A + "print(s.isdecimal())")
case("str.isdigit", A + "print(s.isdigit())")
case("str.isnumeric", A + "print(s.isnumeric())")
case("str.isupper", A + "print(s.isupper())")
case("str.islower", A + "print(s.islower())")
case("str.isprintable", A + "print(s.isprintable())")
case("str.istitle", A + "print(s.istitle())")
case("str.isalnum", A + "print(s.isalnum())")
case("str.isidentifier", A + "print(s.isidentifier())")
case("str.isascii", A + "print(s.isascii())")
case("str.find", A + 'print(s.find("o"))')
case("str.find(s)", A + 'print(s.find("o", 5))')
case("str.find(s,e)", A + 'print(s.find("o", 5, 9))')
case("str.rfind", A + 'print(s.rfind("o"))')
case("str.rfind(s)", A + 'print(s.rfind("o", 5))')
case("str.rfind(s,e)", A + 'print(s.rfind("o", 0, 6))')
case("str.index", A + 'print(s.index("o"))')
case("str.index(s)", A + 'print(s.index("o", 5))')
case("str.index(s,e)", A + 'print(s.index("o", 5, 9))')
case("str.rindex", A + 'print(s.rindex("o"))')
case("str.rindex(s)", A + 'print(s.rindex("o", 5))')
case("str.rindex(s,e)", A + 'print(s.rindex("o", 0, 6))')
case("str.count", A + 'print(s.count("l"))')
case("str.count(s)", A + 'print(s.count("l", 4))')
case("str.count(s,e)", A + 'print(s.count("l", 0, 5))')
case("str.replace", A + 'print(s.replace("l", "L"))')
case("str.replace(n)", A + 'print(s.replace("l", "L", 2))')
case("str.strip", 's: str = "  hi  "\nprint(s.strip())')
case("str.strip(chars)", 's: str = "xxhixx"\nprint(s.strip("x"))')
case("str.lstrip", 's: str = "  hi  "\nprint(s.lstrip() + "|")')
case("str.lstrip(chars)", 's: str = "xxhixx"\nprint(s.lstrip("x"))')
case("str.rstrip", 's: str = "  hi  "\nprint("|" + s.rstrip())')
case("str.rstrip(chars)", 's: str = "xxhixx"\nprint(s.rstrip("x"))')
case("str.removeprefix", A + 'print(s.removeprefix("Hello "))')
case("str.removesuffix", A + 'print(s.removesuffix(" World"))')
case("str.center", A + "print(s.center(15))")
case("str.center(f)", A + 'print(s.center(15, "-"))')
case("str.ljust", A + 'print(s.ljust(15) + "|")')
case("str.ljust(f)", A + 'print(s.ljust(15, "-"))')
case("str.split", 's: str = "a b c"\nprint(s.split())')
case("str.split(sep)", 's: str = "a,b,c"\nprint(s.split(","))')
case("str.split(sep,n)", 's: str = "a,b,c"\nprint(s.split(",", 1))')
case("str.__eq__", A + 'print(s == "Hello World")')
case("str.__ne__", A + 'print(s != "x")')
case("str.__lt__", A + 'print(s < "Z")')
case("str.__le__", A + 'print(s <= "Z")')
case("str.__gt__", A + 'print(s > "Z")')
case("str.__ge__", A + 'print(s >= "Z")')

# ---------------- bytes ----------------
B = 'b: bytes = b"Hello World"\n'
case("bytes.__len__", B + "print(len(b))")
case("bytes.__getitem__", B + "print(b[1])")
case("bytes.__getslice__", B + "print(b[1:4])")
case("bytes.__add__", B + 'print(b + b"!")')
case("bytes.__eq__", B + 'print(b == b"Hello World")')
case("bytes.__ne__", B + 'print(b != b"x")')
case("bytes.__bool__", B + "print(bool(b))")
case("bytes.__repr__", B + "print(repr(b))")
case("bytes.__str__", B + "print(str(b))")
case("bytes.__mul__", 'b: bytes = b"ab"\nprint(b * 2)')
case("bytes.__hash__", B + "print(hash(b) == hash(b))")
case("bytes.decode", B + "print(b.decode())")
case("bytes.split", 'b: bytes = b"a b c"\nprint(b.split())')
case("bytes.split(sep)", 'b: bytes = b"a,b,c"\nprint(b.split(b","))')
case("bytes.find", B + 'print(b.find(b"o"))')
case("bytes.count", B + 'print(b.count(b"l"))')
case("bytes.startswith", B + 'print(b.startswith(b"He"))')
case("bytes.endswith", B + 'print(b.endswith(b"ld"))')
case("bytes.strip", 'b: bytes = b"  hi  "\nprint(b.strip())')
case("bytes.replace", B + 'print(b.replace(b"l", b"L"))')
case("bytes.hex", B + "print(b.hex())")
case("bytes.fromhex", 'print(bytes.fromhex("48656c"))')
case("bytes.join", 'xs: list[bytes] = [b"a", b"b"]\nprint(b",".join(xs))')


def run(cmd, path):
    try:
        r = subprocess.run(cmd + [str(path)], capture_output=True, text=True,
                           timeout=300)
        return r.returncode, r.stdout, r.stderr
    except subprocess.TimeoutExpired:
        return -99, "", "TIMEOUT"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("lyc", type=pathlib.Path)
    ap.add_argument("only", nargs="?", default=None,
                    help="run only cases whose name contains this substring")
    args = ap.parse_args()
    lyc = args.lyc.resolve()

    tmp = pathlib.Path(tempfile.mkdtemp(prefix="contracts-"))
    selected = [(n, s) for n, s in CASES.items()
                if not args.only or args.only in n]
    bad = []
    for name, src in selected:
        p = tmp / "case.py"
        p.write_text(src)
        crc, cout, _ = run([CPY], p)
        lrc, lout, lerr = run([str(lyc), "jit"], p)
        if crc == lrc and cout == lout:
            print(f"ok   {name}")
            continue
        first = (lerr.strip().splitlines() or [""])[0]
        # An MLIR diagnostic is prefixed with a `loc(fused<...>)` blob that is
        # longer than the message; keep the message.
        if "error:" in first:
            first = first[first.index("error:"):]
        print(f"FAIL {name}: cpython(rc={crc}) {cout!r} | "
              f"lyc(rc={lrc}) {lout!r} {first[:200]!r}")
        bad.append(name)
    print(f"\n{len(bad)} disagreements of {len(selected)} cases run")
    for b in bad:
        print("  -", b)
    return len(bad)


if __name__ == "__main__":
    sys.exit(main())
