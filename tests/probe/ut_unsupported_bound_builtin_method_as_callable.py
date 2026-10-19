# WHAT: A builtin container's bound method (`log.append`) passed as a
#   Callable -- `self.addCleanup(log.append, x)` in a test. Refused with
#   "attr.get object type has no class schema".
# KIND: 未対応 (unsupported) -- refused with an internal message
# FOUND: 2026-10-20, porting unittest -- docs/unittest-port.md lists every ut_ probe.
# FIXED: 2026-10-20 -- a method read off a builtin instance is a forwarder over the
#   captured receiver (cases/a_method_of_a_builtin_value_read_without_calling_it).
# CLASSIFICATION @ lang-gaps 3fa9a5fb + the fix: 1 正しい
# CPython 3.14 expects: ['hi']
from collections.abc import Callable


def apply(f: Callable[[str], object], text: str) -> None:
    f(text)


log: list[str] = []
apply(log.append, "hi")
print(log)
