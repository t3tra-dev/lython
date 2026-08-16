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


# --- `raise E` with no call, which is `raise E()` --------------------------
# CPython instantiates a raised CLASS with no arguments. The walk handed the
# type object straight to `py.raise`, which asked the runtime for a `.raise`
# primitive on it -- "runtime manifest has no .raise primitive". Only the
# no-argument spelling was refused; `raise E("x")` was always fine. It is the
# spelling `raise StopIteration` inside a hand-written `__next__` uses, and the
# one every `raise ValueError` guard is written in.
#
# The message matters as much as the class: an instance built with no arguments
# has an EMPTY str, which is what distinguishes this from the argument form.
try:
    raise ValueError
except ValueError as exc:
    print("bare value:", repr(str(exc)))

try:
    raise ConfigError
except AppError as exc:
    print("bare config:", repr(str(exc)))

try:
    raise KeyError
except KeyError as exc:
    print("bare key:", repr(str(exc)))

try:
    raise StopIteration
except StopIteration as exc:
    print("bare stop:", repr(str(exc)))


def guard(k: int) -> int:
    if k < 0:
        raise ValueError
    return k


for k in [-1, 2]:
    try:
        print("guard:", guard(k))
    except ValueError as exc:
        print("guard raised:", repr(str(exc)))
