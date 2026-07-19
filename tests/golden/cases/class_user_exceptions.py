class AppError(Exception):
    pass

class ConfigError(AppError):
    pass

class ParseError(ValueError):
    def __init__(self, detail: str) -> None:
        super().__init__("parse failed: " + detail)

def check(kind: int) -> None:
    if kind == 0:
        raise AppError("app-level")
    if kind == 1:
        raise ConfigError("bad config")
    raise ParseError("line 3")

for k in [0, 1, 2]:
    try:
        check(k)
    except ConfigError as e:
        print("config:", str(e))
    except AppError as e:
        print("app:", str(e))
    except ValueError as e:
        print("value:", str(e))

try:
    raise ConfigError("caught as Exception")
except Exception as e:
    print("exc:", str(e))
