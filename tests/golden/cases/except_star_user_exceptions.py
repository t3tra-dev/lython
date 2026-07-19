class AppError(Exception):
    pass

class DbError(AppError):
    pass

try:
    raise ExceptionGroup("ops", [DbError("db down"), ValueError("v"), AppError("app")])
except* AppError as e:
    print("A:", repr(e))
except* ValueError as e:
    print("V:", repr(e))
print("after")
