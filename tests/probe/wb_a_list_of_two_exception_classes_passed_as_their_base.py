# WHAT: A list displayed from instances of two exception classes,
#   `[E("e"), F("f")]`, iterated and each element passed where BaseException
#   is expected: refused, because the element type the display joins to is not
#   taken as a BaseException. Annotating the list `list[BaseException]` works.
# KIND: 未対応 (unsupported) -- loud
# FOUND: 2026-10-20, writing the format_exception_only golden.
# CLASSIFICATION: loud 拒否 (診断)
#   static type !py.callable<[!py.contract<"builtins.BaseException">], ...> is
#   not callable: call arguments do not match the Callable contract
# CPython 3.14 expects: E('e') / F('f')
class E(Exception):
    pass


class F(Exception):
    pass


def show(e: BaseException) -> None:
    print(repr(e))


for x in [E("e"), F("f")]:
    show(x)
