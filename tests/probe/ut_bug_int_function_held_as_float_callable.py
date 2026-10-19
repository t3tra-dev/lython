# WHAT: An int-returning function stored in a `Callable[[], float]` variable
#   (int is acceptable where float is expected). The program compiles, then
#   the call raises "TypeError: callable target is not available"; CPython
#   prints 1.
# KIND: バグ (bug) -- silent: raises where CPython returns
# FOUND: 2026-10-20, porting unittest -- docs/unittest-port.md lists every ut_ probe.
# differential: expect-wrong the call raises a TypeError the program never asked for
# CLASSIFICATION @ lang-gaps fa5c535a: 2 silent 誤実行
#   cpython='1\n' lyc='TypeError: callable target is not available\n'
# CPython 3.14 expects: 1
from collections.abc import Callable


def one() -> int:
    return 1


f: Callable[[], float] = one
try:
    print(f())
except TypeError as e:
    print("TypeError:", e)
