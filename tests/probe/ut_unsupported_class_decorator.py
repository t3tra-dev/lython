# WHAT: A program function used as a class decorator (`@skip(...)` on a
#   TestCase class).
# KIND: 未対応 (unsupported)
# FOUND: 2026-10-20, porting unittest -- docs/unittest-port.md lists every ut_ probe.
# CLASSIFICATION @ lang-gaps fa5c535a: 3 loud 拒否 (診断)
#   decorator 'mark' is not supported (unrecognized decorators are rejected instead of silently ignored)
# CPython 3.14 expects: marked C / C
def mark[T](cls: type[T]) -> type[T]:
    print("marked", cls.__name__)
    return cls


@mark
class C:
    pass


print(C.__name__)
