# What this pins: an annotated module global of THIS program is invisible
# inside an imported module. Every name below is a local of a stdlib function
# this case then calls -- `key`/`result`/`selected` in `collections.Counter`,
# `entry`/`parts`/`index` in `OrderedDict`, `pos`/`n`/`s`/`k`/`head` in
# `string.Template` -- and each is declared here at a type that method's local
# is not.
#
# Why this needs to run rather than assert on a diagnostic: the failure it
# replaces WAS a diagnostic, but it was reported against a stdlib file the
# program never wrote ("<stdlib>/collections.py:181: 'builtins.str' does not
# provide manifest method '__gt__'"), and the direction that survives a
# type-compatible collision is not a diagnostic at all -- the method reads the
# program's global instead of binding its own local. So the assertion has to be
# that the stdlib answers correctly AND that the globals still hold what this
# module put in them.
#
# Every expected line is python3.14's.

from collections import Counter, OrderedDict
from string import Template

key: int = 11
result: str = "kept"
selected: int = 12
entry: int = 13
parts: int = 14
index: int = 15
pos: float = 3.5
n: int = 16
s: str = "sss"
k: str = "kk"
head: str = "h"
value: str = "v"
i: int = 17
j: int = 18
m: int = 19
out: int = 20
limit: int = 21
elem: str = "e"
nxt: str = "x"
kind: str = "K"
idx: int = 22

counts: Counter[str] = Counter()
counts.update(["a", "b", "a", "c", "a", "b"])
print(counts.most_common(2))
print(counts["a"], counts["c"])

ordered: OrderedDict[str, int] = OrderedDict[str, int]()
ordered["x"] = 1
ordered["y"] = 2
ordered["z"] = 3
print(repr(ordered))
print(len(ordered), ordered["y"])

template: Template = Template("$head/$s/$k")
print(template.substitute({"head": "H", "s": "S", "k": "K"}))

# The globals are untouched by any of it.
print(key, result, selected, entry, parts, index)
print(pos, n, s, k, head, value)
print(i, j, m, out, limit, elem, nxt, kind, idx)
