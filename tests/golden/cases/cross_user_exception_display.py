class AppError(Exception):
    pass


class CodedError(AppError):
    def __init__(self, code: int, detail: str) -> None:
        super().__init__("code " + str(code) + ": " + detail)


try:
    raise AppError("plain failure")
except AppError as e:
    print(e)
    print(str(e))
    print(repr(e))
    print(e.args)

try:
    raise CodedError(7, "disk full")
except Exception as e:
    print(e)
    print(str(e))
    print(repr(e))
    print(e.args)

try:
    raise AppError()
except AppError as e:
    print(e)
    print(repr(e))
    print(e.args)
