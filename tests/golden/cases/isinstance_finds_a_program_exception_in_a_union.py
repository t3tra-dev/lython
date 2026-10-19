# WHAT: `isinstance` against a program's exception class, on a value whose
# static type is a union with a builtin exception member -- a parameter typed
# `BaseException | None`, an element of `list[Exception | None]` -- answers by
# the class of the exception the value holds, and the true branch reads the
# subclass's own field.
#
# WHY THIS IS RUN: the answer is the class of the object the union holds at
# run time, which the static type leaves open.


class AppError(Exception):
    pass


class Detail(AppError):
    def __init__(self, code: int) -> None:
        super().__init__("detail " + str(code))
        self.code = code


def describe(e: BaseException | None) -> str:
    if isinstance(e, Detail):
        return "detail " + str(e.code)
    if isinstance(e, AppError):
        return "app"
    if e is None:
        return "none"
    return "other"


print(describe(Detail(7)), describe(AppError("a")), describe(KeyError("k")),
      describe(None))

held: list[Exception | None] = [Detail(3), None, AppError("b"), ValueError("v")]
for h in held:
    print(isinstance(h, Detail), isinstance(h, AppError), isinstance(h, Exception))
