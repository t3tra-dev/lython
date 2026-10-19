# WHAT: A generic decorator factory that returns ANOTHER factory's result
#   (`skipUnless` returning `skipIf(not condition, reason)`): "this call's
#   result is generic in a type parameter no argument names".
# KIND: 未対応 (unsupported)
# FOUND: 2026-10-20, porting unittest -- docs/unittest-port.md lists every ut_ probe.
# FIXED: 2026-10-20 -- a type parameter only the result names is solved from the
#   type the result is wanted as (the factory's own return annotation).
# CLASSIFICATION @ lang-gaps e3a74bc1 + the fix: 1 正しい
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
