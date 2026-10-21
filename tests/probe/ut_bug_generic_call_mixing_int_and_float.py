# WHAT: `collect(0, 1.5)` for `def collect[T](*items: T) -> list[T]` should
#   solve T as float (int is acceptable where float is expected). It is
#   refused, and the message is the one for a curried decorator factory:
#   "apply the result where it is made -- `collect(...)(f)`".
# KIND: バグ (bug) -- loud, with a misleading message
# FOUND: 2026-10-20, porting unittest -- docs/unittest-port.md lists every ut_ probe.
# FIXED: 2026-10-21 -- a parameter already bound meets the next argument as a
#   parameter, so two types join (a union, along the numeric tower too);
#   a bool against another rung is still refused at the call.
# CLASSIFICATION @ lang-gaps d0cc1f6b + the fix: 1 正しい
# CPython 3.14 expects: [0, 1.5]
def collect[T](*items: T) -> list[T]:
    return list(items)


print(collect(0, 1.5))
