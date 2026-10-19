# WHAT: `add(*xs)` with a list of run-time length filling named positional
#   parameters. A starred sequence of unknown length reaches only *args
#   (typed unpack); the parameters it would fill must be given explicitly.
# KIND: 未対応 (unsupported) -- by design for now
# FOUND: 2026-10-20, porting unittest -- docs/unittest-port.md lists every ut_ probe.
# CLASSIFICATION @ lang-gaps fa5c535a: 3 loud 拒否 (診断)
#   a starred argument whose length is known only at run time can fill only *args here, and its first members would fill 'a' and 'b' of 'add': give them explicitly before it
# CPython 3.14 expects: 3
def add(a: int, b: int) -> int:
    return a + b


xs = [1, 2]
print(add(*xs))
