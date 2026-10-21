# WHAT: An int-returning function stored in a `Callable[[], float]` variable
#   (int is acceptable where float is expected). The program compiles, then
#   the call raises "TypeError: callable target is not available"; CPython
#   prints 1.
# KIND: バグ (bug) -- silent: raises where CPython returns
# FOUND: 2026-10-20, porting unittest -- docs/unittest-port.md lists every ut_ probe.
# FIXED: 2026-10-21 -- refused at the assignment, as every other declared
#   storage is: int and float share no representation, and converting would
#   print 1.0. Now loud; CPython's answer needs a callable that holds either.
# CLASSIFICATION @ lang-gaps 5d8e3215 + the fix: 3 未対応 (宣言した型が格納の型)
# CPython 3.14 expects: 1
from collections.abc import Callable


def one() -> int:
    return 1


f: Callable[[], float] = one
try:
    print(f())
except TypeError as e:
    print("TypeError:", e)
