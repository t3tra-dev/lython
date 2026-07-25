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
seven contracts it covered. The contract-audit track then extended CASES over
the other seventy-two, which is where the rest of the second half of this file
comes from. The full result table lives in rfc/contract-audit.md; read it before
adding cases, because it records which of the remaining disagreements are method
gaps and which are unbound builtin *names* (complex, slice, repr, type) whose
methods work fine when reached another way. Extend CASES rather than starting
over.

    python3 tests/probe/tools/contracts.py ./build/bin/lyc [name-substring]

The gate is EXPECTED_FAILURES below, not a count, and it is checked in both
directions: a disagreement not in the set fails, and a case IN the set that now
agrees also fails, as a stale expectation. Exit code is the number of such
deltas, so the target is always zero and it is always reachable -- unlike
contract_scan.py, whose candidate count can never reach zero because two of its
false-positive mechanisms are permanent.

Both directions are needed because the set is tree-relative. Counting instead
would carry slack over exactly the code most likely to regress: the six methods
below are implemented on kernel/side-defects (a4be8bf), so on a tree containing
that branch all ten cases pass, and a `<= 10` gate would then stay green through
a fresh break in any of them. The second direction turns that same merge into an
explicit "these now pass, shrink the set" failure instead.
"""

import argparse
import pathlib
import subprocess
import sys
import tempfile

CPY = "/opt/homebrew/Frameworks/Python.framework/Versions/3.14/bin/python3.14"

# Cases known to disagree with CPython. Measured on kernel/contract-audit at
# 43e84c4: 501 cases, 431 ok, 70 expected.
#
# TREE-RELATIVE, and grouped by *cause* because the causes retire at different
# times. Do not carry any group forward on faith: a case here that starts
# agreeing is reported as a stale expectation and fails the gate, which is the
# whole point of the set. rfc/contract-audit.md holds the diagnosis for each
# group.
EXPECTED_FAILURES = frozenset({
    # -- Implemented on kernel/side-defects (a4be8bf). On any tree containing
    # that branch these ten pass and belong out of the set.
    "list.pop",
    "list.pop(i)",
    "list.pop(-2)",
    "list.insert",
    "list.insert(-1)",
    "list.insert(big)",
    "tuple.__add__",
    "tuple.__mul__",
    "tuple.count",
    "tuple.index",
    "MutableSequence.insert",          # list.insert reached via the protocol

    # -- Unbound builtin *names*, not missing methods. The contract's methods
    # work; the spelling used to reach them is not bound. All eleven complex
    # methods pass when written as literals (1.0 + 2.0j) -- see the RFC, and do
    # not "fix" these by implementing methods that already exist.
    "complex.__add__", "complex.__add__.explicit",
    "complex.__sub__", "complex.__sub__.explicit",
    "complex.__mul__", "complex.__mul__.explicit",
    "complex.__truediv__", "complex.__truediv__.explicit",
    "complex.__neg__", "complex.__neg__.explicit",
    "complex.__pos__", "complex.__pos__.explicit",
    "complex.__eq__", "complex.__eq__.explicit",
    "complex.__ne__", "complex.__ne__.explicit",
    "complex.__repr__", "complex.__repr__.explicit",
    "complex.__str__", "complex.__str__.explicit",
    "complex.__abs__", "complex.__abs__.explicit",
    "slice.indices",                   # `slice` unbound; xs[1:3] works
    "object.__repr__", "object.__str__",   # `repr` unbound on user classes
    "type.__repr__",                       # ditto
    "type.__name__",                       # no static __name__ attribute
    "int.__new__(bytes)",              # int(b"12") overload not declared
    "float.__new__",                   # float("1.5") overload not declared

    # -- builtins.object's default dunders do not reach user classes. Needs a
    # decision about how they are inherited, not a manifest patch.
    "object.__bool__",
    "object.__eq__", "object.__eq__(self)", "object.__eq__.explicit",
    "object.__ne__", "object.__ne__.explicit",
    "object.__hash__", "object.__hash__.explicit",
    "object.__repr__.explicit", "object.__str__.explicit",
    "object.__init__.explicit",

    # -- Declared with no implementation behind them.
    "BaseException.with_traceback",     # wants tracebacks modelled first
    "BaseException.add_note",           # wants a __notes__ field
    "dict_keys.__reversed__",
    "dict_values.__reversed__",
    "dict_items.__reversed__",
    "type.__call__",                   # now diagnosed, still unimplemented
    "type.__or__",

    # -- _asyncio.Future cannot be instantiated ("class instantiation leaves
    # unbound static type parameters"), which hides all 14 of its declarations;
    # Task's are behind the unbound asyncio.ensure_future.
    "Future.set_result", "Future.done", "Future.cancelled", "Future.cancel",
    "Future.__await__",
    "Task.get_name", "Task.set_name", "Task.cancelling", "Task.uncancel",

    # -- Behaviour differs from CPython 3.14 rather than failing.
    "range.__repr__",                  # <range object at ...> vs range(0, 5)
    "nullcontext",                     # __enter__ yields self, not None
    "ContextManager.__enter__",        # __exit__ return type rejected
})

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


# ======================= second pass: the remaining 72 =======================
# Two spellings per dunder where they differ in what they exercise: the operator
# form routes through the emitter's binary/unary op path, the explicit call is
# what `method_names` literally promises. A contract that declares the name owes
# both, and only the explicit form catches a name that no lowering reaches.

# ---------------- int ----------------
I = "i: int = 7\nj: int = 3\n"
case("int.__new__", 'print(int("12"))')
case("int.__new__(float)", "print(int(3.9))")
case("int.__new__(bytes)", 'print(int(b"12"))')
case("int.__new__()", "print(int())")
case("int.__add__", I + "print(i + j)")
case("int.__add__.explicit", I + "print(i.__add__(j))")
case("int.__sub__", I + "print(i - j)")
case("int.__sub__.explicit", I + "print(i.__sub__(j))")
case("int.__mul__", I + "print(i * j)")
case("int.__mul__.explicit", I + "print(i.__mul__(j))")
case("int.__floordiv__", I + "print(i // j)")
case("int.__floordiv__.explicit", I + "print(i.__floordiv__(j))")
case("int.__floordiv__(neg)", "print(-7 // 3)")
case("int.__truediv__", I + "print(i / j)")
case("int.__truediv__.explicit", I + "print(i.__truediv__(j))")
case("int.__mod__", I + "print(i % j)")
case("int.__mod__.explicit", I + "print(i.__mod__(j))")
case("int.__mod__(neg)", "print(-7 % 3)")
case("int.__and__", I + "print(i & j)")
case("int.__and__.explicit", I + "print(i.__and__(j))")
case("int.__or__", I + "print(i | j)")
case("int.__or__.explicit", I + "print(i.__or__(j))")
case("int.__xor__", I + "print(i ^ j)")
case("int.__xor__.explicit", I + "print(i.__xor__(j))")
case("int.__lshift__", I + "print(i << j)")
case("int.__lshift__.explicit", I + "print(i.__lshift__(j))")
case("int.__rshift__", I + "print(i >> j)")
case("int.__rshift__.explicit", I + "print(i.__rshift__(j))")
case("int.__neg__", I + "print(-i)")
case("int.__neg__.explicit", I + "print(i.__neg__())")
case("int.__pos__", I + "print(+i)")
case("int.__pos__.explicit", I + "print(i.__pos__())")
case("int.__invert__", I + "print(~i)")
case("int.__invert__.explicit", I + "print(i.__invert__())")
case("int.__round__", I + "print(round(i))")
case("int.__round__(n)", I + "print(round(i, 1))")
case("int.__round__.explicit", I + "print(i.__round__(1))")
case("int.__int__", I + "print(int(i))")
case("int.__int__.explicit", I + "print(i.__int__())")
case("int.__float__", I + "print(float(i))")
case("int.__float__.explicit", I + "print(i.__float__())")
case("int.__bool__", I + "print(bool(i))")
case("int.__bool__.explicit", I + "print(i.__bool__())")
case("int.__index__.explicit", I + "print(i.__index__())")
case("int.__hash__", I + "print(hash(i))")
case("int.__hash__.explicit", I + "print(i.__hash__())")
case("int.__lt__", I + "print(i < j)")
case("int.__lt__.explicit", I + "print(i.__lt__(j))")
case("int.__le__", I + "print(i <= j)")
case("int.__le__.explicit", I + "print(i.__le__(j))")
case("int.__gt__", I + "print(i > j)")
case("int.__gt__.explicit", I + "print(i.__gt__(j))")
case("int.__ge__", I + "print(i >= j)")
case("int.__ge__.explicit", I + "print(i.__ge__(j))")
case("int.__repr__", I + "print(repr(i))")
case("int.__repr__.explicit", I + "print(i.__repr__())")
case("int.__str__", I + "print(str(i))")
case("int.__str__.explicit", I + "print(i.__str__())")
case("int.__eq__", I + "print(i == j)")
case("int.__eq__.explicit", I + "print(i.__eq__(j))")
case("int.__ne__", I + "print(i != j)")
case("int.__ne__.explicit", I + "print(i.__ne__(j))")
case("int.__pow__", I + "print(i ** j)")
case("int.__pow__.explicit", I + "print(i.__pow__(j))")
case("int.__abs__", "i: int = -7\nprint(abs(i))")
case("int.__abs__.explicit", "i: int = -7\nprint(i.__abs__())")
case("int.__format__", I + 'print(format(i, "d"))')
case("int.__format__.explicit", I + 'print(i.__format__("d"))')
case("int.__lt__(float)", "i: int = 7\nprint(i < 7.5)")
case("int.__le__(float)", "i: int = 7\nprint(i <= 7.5)")
case("int.__gt__(float)", "i: int = 7\nprint(i > 7.5)")
case("int.__ge__(float)", "i: int = 7\nprint(i >= 7.5)")
case("int.__eq__(float)", "i: int = 7\nprint(i == 7.0)")
case("int.__ne__(float)", "i: int = 7\nprint(i != 7.0)")

# ---------------- float ----------------
X = "x: float = 7.5\ny: float = 2.0\n"
case("float.__new__", 'print(float("1.5"))')
case("float.__new__(int)", "print(float(3))")
case("float.__new__()", "print(float())")
case("float.__repr__", X + "print(repr(x))")
case("float.__repr__.explicit", X + "print(x.__repr__())")
case("float.__add__", X + "print(x + y)")
case("float.__add__.explicit", X + "print(x.__add__(y))")
case("float.__sub__", X + "print(x - y)")
case("float.__sub__.explicit", X + "print(x.__sub__(y))")
case("float.__mul__", X + "print(x * y)")
case("float.__mul__.explicit", X + "print(x.__mul__(y))")
case("float.__truediv__", X + "print(x / y)")
case("float.__truediv__.explicit", X + "print(x.__truediv__(y))")
case("float.__floordiv__", X + "print(x // y)")
case("float.__floordiv__.explicit", X + "print(x.__floordiv__(y))")
case("float.__floordiv__(neg)", "print(-7.5 // 2.0)")
case("float.__mod__", X + "print(x % y)")
case("float.__mod__.explicit", X + "print(x.__mod__(y))")
case("float.__mod__(neg)", "print(-7.5 % 2.0)")
case("float.__float__", X + "print(float(x))")
case("float.__float__.explicit", X + "print(x.__float__())")
case("float.__bool__", X + "print(bool(x))")
case("float.__bool__.explicit", X + "print(x.__bool__())")
case("float.__round__", X + "print(round(x))")
case("float.__round__(n)", "x: float = 3.14159\nprint(round(x, 2))")
case("float.__round__.explicit", "x: float = 3.14159\nprint(x.__round__(2))")
case("float.__round__(half)", "print(round(2.5))\nprint(round(3.5))\nprint(round(-2.5))")
case("float.__lt__", X + "print(x < y)")
case("float.__lt__.explicit", X + "print(x.__lt__(y))")
case("float.__le__", X + "print(x <= y)")
case("float.__le__.explicit", X + "print(x.__le__(y))")
case("float.__gt__", X + "print(x > y)")
case("float.__gt__.explicit", X + "print(x.__gt__(y))")
case("float.__ge__", X + "print(x >= y)")
case("float.__ge__.explicit", X + "print(x.__ge__(y))")
case("float.__str__", X + "print(str(x))")
case("float.__str__.explicit", X + "print(x.__str__())")
case("float.__eq__", X + "print(x == y)")
case("float.__eq__.explicit", X + "print(x.__eq__(y))")
case("float.__ne__", X + "print(x != y)")
case("float.__ne__.explicit", X + "print(x.__ne__(y))")
case("float.__pow__", X + "print(x ** y)")
case("float.__pow__.explicit", X + "print(x.__pow__(y))")
case("float.__hash__", X + "print(hash(x))")
case("float.__hash__.explicit", X + "print(x.__hash__())")
case("float.__abs__", "x: float = -7.5\nprint(abs(x))")
case("float.__abs__.explicit", "x: float = -7.5\nprint(x.__abs__())")
case("float.__format__", X + 'print(format(x, ".2f"))')
case("float.__format__.explicit", X + 'print(x.__format__(".2f"))')
case("float.__lt__(int)", "x: float = 7.5\nprint(x < 8)")
case("float.__le__(int)", "x: float = 7.5\nprint(x <= 8)")
case("float.__gt__(int)", "x: float = 7.5\nprint(x > 8)")
case("float.__ge__(int)", "x: float = 7.5\nprint(x >= 8)")
case("float.__eq__(int)", "x: float = 7.0\nprint(x == 7)")
case("float.__ne__(int)", "x: float = 7.0\nprint(x != 7)")
case("float.__neg__", X + "print(-x)")
case("float.__neg__.explicit", X + "print(x.__neg__())")
case("float.__pos__", X + "print(+x)")
case("float.__pos__.explicit", X + "print(x.__pos__())")

# ---------------- bool ----------------
case("bool.__new__", "print(bool(1))")
case("bool.__new__(str)", 'print(bool(""))')
case("bool.__new__()", "print(bool())")
case("bool.__repr__", "b: bool = True\nprint(repr(b))")
case("bool.__repr__.explicit", "b: bool = True\nprint(b.__repr__())")
case("bool.__str__", "b: bool = True\nprint(str(b))")
case("bool.__str__.explicit", "b: bool = True\nprint(b.__str__())")
case("bool.__bool__.explicit", "b: bool = True\nprint(b.__bool__())")
case("bool.__and__", "b: bool = True\nc: bool = False\nprint(b & c)")
case("bool.__and__.explicit", "b: bool = True\nc: bool = False\nprint(b.__and__(c))")
case("bool.__or__", "b: bool = True\nc: bool = False\nprint(b | c)")
case("bool.__or__.explicit", "b: bool = True\nc: bool = False\nprint(b.__or__(c))")
case("bool.__xor__", "b: bool = True\nc: bool = False\nprint(b ^ c)")
case("bool.__xor__.explicit", "b: bool = True\nc: bool = False\nprint(b.__xor__(c))")
case("bool.__hash__", "b: bool = True\nprint(hash(b))")
case("bool.__hash__.explicit", "b: bool = True\nprint(b.__hash__())")
case("bool.__format__", "b: bool = True\nprint(format(b, \"\"))")
case("bool.__format__.explicit", "b: bool = True\nprint(b.__format__(\"\"))")

# ---------------- complex ----------------
C = "u: complex = complex(1.0, 2.0)\nv: complex = complex(3.0, -1.0)\n"
case("complex.__add__", C + "print(u + v)")
case("complex.__add__.explicit", C + "print(u.__add__(v))")
case("complex.__sub__", C + "print(u - v)")
case("complex.__sub__.explicit", C + "print(u.__sub__(v))")
case("complex.__mul__", C + "print(u * v)")
case("complex.__mul__.explicit", C + "print(u.__mul__(v))")
case("complex.__truediv__", C + "print(u / v)")
case("complex.__truediv__.explicit", C + "print(u.__truediv__(v))")
case("complex.__neg__", C + "print(-u)")
case("complex.__neg__.explicit", C + "print(u.__neg__())")
case("complex.__pos__", C + "print(+u)")
case("complex.__pos__.explicit", C + "print(u.__pos__())")
case("complex.__eq__", C + "print(u == v)")
case("complex.__eq__.explicit", C + "print(u.__eq__(v))")
case("complex.__ne__", C + "print(u != v)")
case("complex.__ne__.explicit", C + "print(u.__ne__(v))")
case("complex.__repr__", C + "print(repr(u))")
case("complex.__repr__.explicit", C + "print(u.__repr__())")
case("complex.__str__", C + "print(str(u))")
case("complex.__str__.explicit", C + "print(u.__str__())")
case("complex.__abs__", C + "print(abs(u))")
case("complex.__abs__.explicit", C + "print(u.__abs__())")

# ---------------- object ----------------
OB = "class Box:\n    def __init__(self) -> None:\n        self.v: int = 1\nb = Box()\n"
case("object.__init__", OB + "print(b.v)")
case("object.__repr__", OB + "print(repr(b).startswith('<'))")
case("object.__str__", OB + "print(str(b).startswith('<'))")
case("object.__bool__", OB + "print(bool(b))")
case("object.__eq__", OB + "c = Box()\nprint(b == c)")
case("object.__eq__(self)", OB + "print(b == b)")
case("object.__ne__", OB + "c = Box()\nprint(b != c)")
case("object.__hash__", OB + "print(hash(b) == hash(b))")
case("object.__getattribute__", OB + "print(b.v)")
case("object.__setattr__", OB + "b.v = 5\nprint(b.v)")
case("object.__repr__.explicit", OB + "print(b.__repr__().startswith('<'))")
case("object.__str__.explicit", OB + "print(b.__str__().startswith('<'))")
case("object.__eq__.explicit", OB + "c = Box()\nprint(b.__eq__(c))")
case("object.__ne__.explicit", OB + "c = Box()\nprint(b.__ne__(c))")
case("object.__hash__.explicit", OB + "print(b.__hash__() == b.__hash__())")
case("object.__init__.explicit", OB + "print(Box.__name__)")

# ---------------- range / slice / iterators ----------------
case("range.__len__", "r = range(5)\nprint(len(r))")
case("range.__getitem__", "r = range(5)\nprint(r[2])")
case("range.__contains__", "r = range(5)\nprint(3 in r)")
case("range.__iter__", "t: int = 0\nfor v in range(4):\n    t = t + v\nprint(t)")
case("range.__repr__", "r = range(5)\nprint(repr(r))")
case("range_iterator.__next__", "it = iter(range(3))\nprint(next(it))")
case("range_iterator.__iter__", "it = iter(range(3))\nprint(next(iter(it)))")
case("str_iterator.__next__", 'it = iter("ab")\nprint(next(it))')
case("str_iterator.__iter__", 'it = iter("ab")\nprint(next(iter(it)))')
case("slice.indices", "s = slice(1, 5, 2)\nprint(s.indices(10))")

# ---------------- dict views ----------------
DV = 'd: dict[str, int] = {"a": 1, "b": 2}\n'
case("dict_keys.__iter__", DV + "n: int = 0\nfor k in d.keys():\n    n = n + 1\nprint(n)")
case("dict_keys.__len__", DV + "print(len(d.keys()))")
case("dict_keys.__contains__", DV + 'print("a" in d.keys())')
case("dict_keys.__reversed__", DV + "ks = reversed(d.keys())\nprint(next(ks))")
case("dict_values.__iter__", DV + "t: int = 0\nfor v in d.values():\n    t = t + v\nprint(t)")
case("dict_values.__len__", DV + "print(len(d.values()))")
case("dict_values.__reversed__", DV + "vs = reversed(d.values())\nprint(next(vs))")
case("dict_items.__iter__", DV + "t: int = 0\nfor k, v in d.items():\n    t = t + v\nprint(t)")
case("dict_items.__len__", DV + "print(len(d.items()))")
case("dict_items.__reversed__", DV + "its = reversed(d.items())\nk, v = next(its)\nprint(v)")

# ---------------- BaseException ----------------
case("BaseException.__init__", 'e = ValueError("m")\nprint(e)')
case("BaseException.__str__", 'e = ValueError("m")\nprint(str(e))')
case("BaseException.__repr__", 'e = ValueError("m")\nprint(repr(e))')
case("BaseException.args", 'e = ValueError("m")\nprint(e.args)')
case("BaseException.with_traceback",
     'e = ValueError("m")\nf = e.with_traceback(None)\nprint(str(f))')
case("BaseException.add_note",
     'e = ValueError("m")\ne.add_note("hint")\nprint(str(e))')

# ---------------- type ----------------
case("type.__call__", "t = int\nprint(t())")
case("type.__name__", "print(int.__name__)")
case("type.__or__", "print(int | None)")
case("type.__repr__", "print(repr(int))")

# ---------------- generator / coroutine ----------------
GEN = ("from typing import Generator\n"
       "def g() -> Generator[int, None, None]:\n"
       "    yield 1\n    yield 2\n")
case("GeneratorType.__next__", GEN + "it = g()\nprint(next(it))")
case("GeneratorType.__iter__", GEN + "t: int = 0\nfor v in g():\n    t = t + v\nprint(t)")
case("GeneratorType.send", GEN + "it = g()\nprint(next(it))\nprint(it.send(None))")
case("GeneratorType.close", GEN + "it = g()\nprint(next(it))\nit.close()\nprint(1)")

# ---------------- _io.StringIO ----------------
SIO = "from io import StringIO\n"
case("StringIO.__init__", SIO + 's = StringIO("hi")\nprint(s.getvalue())')
case("StringIO.__init__()", SIO + "s = StringIO()\nprint(s.getvalue() == '')")
case("StringIO.write", SIO + 's = StringIO()\nprint(s.write("ab"))')
case("StringIO.getvalue", SIO + 's = StringIO()\ns.write("ab")\nprint(s.getvalue())')
case("StringIO.read", SIO + 's = StringIO("abc")\nprint(s.read())')
case("StringIO.read(n)", SIO + 's = StringIO("abc")\nprint(s.read(2))')
case("StringIO.seek", SIO + 's = StringIO("abc")\nprint(s.seek(1))\nprint(s.read())')
case("StringIO.seek(whence)", SIO + 's = StringIO("abc")\nprint(s.seek(0, 2))')
case("StringIO.tell", SIO + 's = StringIO("abc")\ns.read(2)\nprint(s.tell())')
case("StringIO.truncate", SIO + 's = StringIO("abc")\nprint(s.truncate(1))\nprint(s.getvalue())')
case("StringIO.truncate()", SIO + 's = StringIO("abc")\ns.seek(1)\nprint(s.truncate())')
case("StringIO.seekable", SIO + "s = StringIO()\nprint(s.seekable())")
case("StringIO.readable", SIO + "s = StringIO()\nprint(s.readable())")
case("StringIO.writable", SIO + "s = StringIO()\nprint(s.writable())")
case("StringIO.close", SIO + "s = StringIO()\ns.close()\nprint(1)")

# ---------------- _io.BytesIO ----------------
BIO = "from io import BytesIO\n"
case("BytesIO.__init__", BIO + 'b = BytesIO(b"hi")\nprint(b.getvalue())')
case("BytesIO.__init__()", BIO + 'b = BytesIO()\nprint(b.getvalue() == b"")')
case("BytesIO.write", BIO + 'b = BytesIO()\nprint(b.write(b"ab"))')
case("BytesIO.getvalue", BIO + 'b = BytesIO()\nb.write(b"ab")\nprint(b.getvalue())')
case("BytesIO.read", BIO + 'b = BytesIO(b"abc")\nprint(b.read())')
case("BytesIO.read(n)", BIO + 'b = BytesIO(b"abc")\nprint(b.read(2))')
case("BytesIO.seek", BIO + 'b = BytesIO(b"abc")\nprint(b.seek(1))\nprint(b.read())')
case("BytesIO.seek(whence)", BIO + 'b = BytesIO(b"abc")\nprint(b.seek(0, 2))')
case("BytesIO.tell", BIO + 'b = BytesIO(b"abc")\nb.read(2)\nprint(b.tell())')
case("BytesIO.truncate", BIO + 'b = BytesIO(b"abc")\nprint(b.truncate(1))\nprint(b.getvalue())')
case("BytesIO.truncate()", BIO + 'b = BytesIO(b"abc")\nb.seek(1)\nprint(b.truncate())')
case("BytesIO.seekable", BIO + "b = BytesIO()\nprint(b.seekable())")
case("BytesIO.readable", BIO + "b = BytesIO()\nprint(b.readable())")
case("BytesIO.writable", BIO + "b = BytesIO()\nprint(b.writable())")
case("BytesIO.close", BIO + "b = BytesIO()\nb.close()\nprint(1)")

# ---------------- _io.TextIOWrapper (sys.stdout) ----------------
case("TextIOWrapper.write", "import sys\nn: int = sys.stdout.write('x\\n')\nprint(n)")
case("TextIOWrapper.flush", "import sys\nsys.stdout.flush()\nprint(1)")
case("TextIOWrapper.fileno", "import sys\nprint(sys.stdout.fileno())")
case("TextIOWrapper.writable", "import sys\nprint(sys.stdout.writable())")
case("TextIOWrapper.readable", "import sys\nprint(sys.stdout.readable())")
case("TextIOWrapper.seekable", "import sys\nprint(sys.stdout.seekable())")

# ---------------- asyncio Future / Task ----------------
AS = "import asyncio\n"
case("Future.set_result",
     AS + "async def m() -> int:\n"
     "    f: asyncio.Future[int] = asyncio.Future()\n"
     "    f.set_result(5)\n    return f.result()\n"
     "print(asyncio.run(m()))")
case("Future.done",
     AS + "async def m() -> bool:\n"
     "    f: asyncio.Future[int] = asyncio.Future()\n"
     "    return f.done()\n"
     "print(asyncio.run(m()))")
case("Future.cancelled",
     AS + "async def m() -> bool:\n"
     "    f: asyncio.Future[int] = asyncio.Future()\n"
     "    return f.cancelled()\n"
     "print(asyncio.run(m()))")
case("Future.cancel",
     AS + "async def m() -> bool:\n"
     "    f: asyncio.Future[int] = asyncio.Future()\n"
     "    return f.cancel()\n"
     "print(asyncio.run(m()))")
case("Future.__await__",
     AS + "async def m() -> int:\n"
     "    f: asyncio.Future[int] = asyncio.Future()\n"
     "    f.set_result(7)\n    return await f\n"
     "print(asyncio.run(m()))")
case("Task.get_name",
     AS + "async def w() -> int:\n    return 1\n"
     "async def m() -> str:\n"
     "    t = asyncio.ensure_future(w())\n"
     "    n: str = t.get_name()\n    await t\n    return n\n"
     "print(asyncio.run(m()).startswith('Task-'))")
case("Task.set_name",
     AS + "async def w() -> int:\n    return 1\n"
     "async def m() -> str:\n"
     "    t = asyncio.ensure_future(w())\n"
     "    t.set_name('x')\n    await t\n    return t.get_name()\n"
     "print(asyncio.run(m()))")
case("Task.cancelling",
     AS + "async def w() -> int:\n    return 1\n"
     "async def m() -> int:\n"
     "    t = asyncio.ensure_future(w())\n"
     "    await t\n    return t.cancelling()\n"
     "print(asyncio.run(m()))")
case("Task.uncancel",
     AS + "async def w() -> int:\n    return 1\n"
     "async def m() -> int:\n"
     "    t = asyncio.ensure_future(w())\n"
     "    await t\n    return t.uncancel()\n"
     "print(asyncio.run(m()))")

# ---------------- contextlib.nullcontext ----------------
case("nullcontext",
     "from contextlib import nullcontext\n"
     "with nullcontext() as c:\n    print(c)")

# ---------------- types.NoneType / misc ----------------
case("NoneType.__bool__", "n = None\nprint(bool(n))")
case("NoneType.__str__", "n = None\nprint(str(n))")
case("SupportsInt.__int__", "print(int(3.5))")
case("SupportsFloat.__float__", "print(float(3))")
case("SupportsIndex.__index__", "xs: list[int] = [1, 2, 3]\nprint(xs[2])")

# ---------------- typing.ContextManager ----------------
case("ContextManager.__enter__",
     "class Res:\n"
     "    def __enter__(self) -> int:\n        return 5\n"
     "    def __exit__(self, a: object, b: object, c: object) -> None:\n"
     "        return None\n"
     "with Res() as v:\n    print(v)")

# ---------------- collections.abc protocol tower (via concretes) ----------
case("Sized.__len__", 'print(len("abc"))')
case("Container.__contains__", "xs: list[int] = [1, 2]\nprint(2 in xs)")
case("Iterable.__iter__", "xs: list[int] = [1, 2]\nprint(sum(x for x in xs) if False else len(xs))")
case("Sequence.count", "xs: list[int] = [1, 1, 2]\nprint(xs.count(1))")
case("Sequence.index", "xs: list[int] = [1, 1, 2]\nprint(xs.index(2))")
case("MutableSequence.insert", "xs: list[int] = [1, 2]\nxs.insert(0, 9)\nprint(xs)")
case("Mapping.get", 'd: dict[str, int] = {"a": 1}\nprint(d.get("a", 0))')

# ---------------- lyrt awaitables / counters ----------------
case("lyrt.Counter", "import lyrt\nc = lyrt.Counter(3)\nprint(next(iter(c)))")

# ---------------- range __new__ overloads ----------------
case("range.__new__(stop)", "print(len(range(3)))")
case("range.__new__(start,stop)", "print(len(range(1, 4)))")
case("range.__new__(start,stop,step)", "print(len(range(1, 7, 2)))")


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
    new_failures, stale, expected = [], [], []
    for name, src in selected:
        p = tmp / "case.py"
        p.write_text(src)
        crc, cout, _ = run([CPY], p)
        lrc, lout, lerr = run([str(lyc), "jit"], p)
        agrees = (crc == lrc and cout == lout)

        if agrees and name not in EXPECTED_FAILURES:
            print(f"ok    {name}")
        elif agrees:
            # Direction two: something got implemented. A count-based gate
            # cannot see this, which is why it silently accrues slack.
            print(f"FIXED {name}  <- remove from EXPECTED_FAILURES")
            stale.append(name)
        else:
            first = (lerr.strip().splitlines() or [""])[0]
            # An MLIR diagnostic is prefixed with a `loc(fused<...>)` blob
            # longer than the message; keep the message.
            if "error:" in first:
                first = first[first.index("error:"):]
            detail = (f"cpython(rc={crc}) {cout!r} | lyc(rc={lrc}) {lout!r} "
                      f"{first[:200]!r}")
            if name in EXPECTED_FAILURES:
                print(f"xfail {name}")
                expected.append(name)
            else:
                print(f"FAIL  {name}: {detail}")
                new_failures.append(name)

    print(f"\n{len(selected)} cases run: "
          f"{len(selected) - len(new_failures) - len(stale) - len(expected)} ok, "
          f"{len(expected)} expected failures, "
          f"{len(new_failures)} new failures, {len(stale)} stale expectations")
    for n in new_failures:
        print("  new failure:      ", n)
    for n in stale:
        print("  stale expectation:", n)
    return len(new_failures) + len(stale)


if __name__ == "__main__":
    sys.exit(main())
