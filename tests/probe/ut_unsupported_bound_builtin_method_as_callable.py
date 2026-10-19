# WHAT: A builtin container's bound method (`log.append`) passed as a
#   Callable -- `self.addCleanup(log.append, x)` in a test. Refused with
#   "attr.get object type has no class schema".
# KIND: 未対応 (unsupported) -- refused with an internal message
# FOUND: 2026-10-20, porting unittest -- docs/unittest-port.md lists every ut_ probe.
# CLASSIFICATION @ lang-gaps fa5c535a: 3 loud 拒否 (診断)
#   attr.get object type has no class schema
# CPython 3.14 expects: ['hi']
from collections.abc import Callable


def apply(f: Callable[[str], object], text: str) -> None:
    f(text)


log: list[str] = []
apply(log.append, "hi")
print(log)
