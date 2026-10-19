# WHAT: A generic decorator factory that returns ANOTHER factory's result
#   (`skipUnless` returning `skipIf(not condition, reason)`): "this call's
#   result is generic in a type parameter no argument names".
# KIND: 未対応 (unsupported)
# FOUND: 2026-10-20, porting unittest -- docs/unittest-port.md lists every ut_ probe.
# CLASSIFICATION @ lang-gaps fa5c535a: 3 loud 拒否 (診断)
#   this call's result is generic in a type parameter no argument names: apply the result where it is made -- `skipIf(...)(f)`, as a decorator does -- which is what solves it
# CPython 3.14 expects: skipped: why
from collections.abc import Callable


def skipIf[T](condition: bool, reason: str) -> Callable[[Callable[[T], None]], Callable[[T], None]]:
    def decorator(f: Callable[[T], None]) -> Callable[[T], None]:
        if condition:
            def skipped(self: T) -> None:
                print("skipped:", reason)
            return skipped
        return f
    return decorator


def skipUnless[T](condition: bool, reason: str) -> Callable[[Callable[[T], None]], Callable[[T], None]]:
    return skipIf(not condition, reason)


class C:
    @skipUnless(False, "why")
    def test(self) -> None:
        print("ran")


C().test()
