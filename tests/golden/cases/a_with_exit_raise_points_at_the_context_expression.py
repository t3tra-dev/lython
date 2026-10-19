# An exception raised by `__exit__` is reported at the `with` item's context
# expression, as CPython locates the exit call: the caret line goes under
# `refuse_quietly(1)` with a call's markers. It went under the first
# characters of the `with` statement. unittest's assertRaises reports
# "KeyError not raised" from its `__exit__`.
# Why run it: the carets are drawn from the location the run records.
import sys
import traceback


class Refusing:
    def __init__(self, code: int) -> None:
        self.code = code

    def __enter__(self) -> "Refusing":
        return self

    def __exit__(self, typ: type[BaseException] | None, value: BaseException | None,
                 tb: object) -> bool:
        raise ValueError("refused " + str(self.code))


def refuse_quietly(code: int) -> Refusing:
    return Refusing(code)


def body() -> None:
    with refuse_quietly(1):
        pass


try:
    body()
except ValueError:
    for line in traceback.format_exc().splitlines():
        if not line.startswith("  File "):
            print(line)
