# WHAT: `add(**d)` with a dict of run-time keys filling named parameters: a
#   mapping of unknown keys reaches only **kwargs.
# KIND: 未対応 (unsupported) -- by design for now
# FOUND: 2026-10-20, porting unittest -- docs/unittest-port.md lists every ut_ probe.
# CLASSIFICATION @ lang-gaps fa5c535a: 3 loud 拒否 (診断)
#   a `**` mapping whose keys are known only at run time can fill only **kwargs here, and its keys could name 'a' and 'b' of 'add', which Python binds first: give them explicitly
# CPython 3.14 expects: 3
def add(a: int, b: int) -> int:
    return a + b


d = {"a": 1, "b": 2}
print(add(**d))
