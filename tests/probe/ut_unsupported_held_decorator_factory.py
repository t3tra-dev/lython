# WHAT: A decorator factory's result held in a name (`t = tag("x")`) and
#   applied with `@t`; the factory applied in place (`@tag("x")`) works.
# KIND: 未対応 (unsupported)
# FOUND: 2026-10-20, porting unittest -- docs/unittest-port.md lists every ut_ probe.
# CLASSIFICATION @ lang-gaps fa5c535a: 3 loud 拒否 (診断)
#   decorator 't' is not supported (unrecognized decorators are rejected instead of silently ignored)
# CPython 3.14 expects: tagged x / 1
from collections.abc import Callable


def tag[F](label: str) -> Callable[[F], F]:
    def apply(f: F) -> F:
        print("tagged", label)
        return f
    return apply


t = tag("x")


@t
def f() -> int:
    return 1


print(f())
