# WHAT: `collect(0, 1.5)` for `def collect[T](*items: T) -> list[T]` should
#   solve T as float (int is acceptable where float is expected). It is
#   refused, and the message is the one for a curried decorator factory:
#   "apply the result where it is made -- `collect(...)(f)`".
# KIND: バグ (bug) -- loud, with a misleading message
# FOUND: 2026-10-20, porting unittest -- docs/unittest-port.md lists every ut_ probe.
# CLASSIFICATION @ lang-gaps fa5c535a: 3 loud 拒否 (診断)
#   this call's result is generic in a type parameter no argument names: apply the result where it is made -- `collect(...)(f)`, as a decorator does -- which is what solves it
# CPython 3.14 expects: [0, 1.5]
def collect[T](*items: T) -> list[T]:
    return list(items)


print(collect(0, 1.5))
