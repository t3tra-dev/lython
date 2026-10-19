# WHAT: A program decorator under @staticmethod (also: on classmethods,
#   properties, generators and methods with packs or keyword-only
#   parameters).
# KIND: 未対応 (unsupported)
# FOUND: 2026-10-20, porting unittest -- docs/unittest-port.md lists every ut_ probe.
# CLASSIFICATION @ lang-gaps fa5c535a: 3 loud 拒否 (診断)
#   decorator 'logged' is not supported (unrecognized decorators are rejected instead of silently ignored)
# CPython 3.14 expects: call 3 / 6
from collections.abc import Callable


def logged(f: Callable[[int], int]) -> Callable[[int], int]:
    def inner(x: int) -> int:
        print("call", x)
        return f(x)
    return inner


class C:
    @staticmethod
    @logged
    def twice(x: int) -> int:
        return 2 * x


print(C.twice(3))
