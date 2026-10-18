# WHAT: `except t:` where `t` is a class value -- a parameter, a field, a loop
# variable over a list of classes -- catches what the class the value names
# would catch, and falls through to the next handler otherwise; `isinstance`
# against the same value agrees.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: which handler runs is the
# run-time match of the raised exception against the class the value holds.
# The handler used to match the value's STATIC bound, so an
# `expected: type[BaseException]` caught every exception -- an assertRaises
# that passes on the wrong exception -- and printed "caught" where CPython
# falls through.
def catches(exc: type[BaseException]) -> str:
    try:
        raise KeyError("k")
    except exc:
        return "caught"
    except BaseException:
        return "fell through"


print(catches(LookupError), catches(ValueError), catches(KeyError))


class Expect:
    def __init__(self, expected: type[BaseException]) -> None:
        self.expected = expected

    def run(self, n: int) -> str:
        try:
            if n == 0:
                raise KeyError("k")
            raise ValueError("v")
        except self.expected:
            return "expected " + self.expected.__name__
        except Exception as e:
            return "other " + type(e).__name__

    def matches(self, e: BaseException) -> bool:
        return isinstance(e, self.expected)


print(Expect(KeyError).run(0), Expect(KeyError).run(1), Expect(LookupError).run(0))
print(Expect(KeyError).matches(ValueError("v")), Expect(Exception).matches(KeyError("k")))


def check(exc: type[Exception], raises: int) -> str:
    try:
        if raises == 0:
            raise KeyError("k")
        if raises == 1:
            raise ValueError("v")
        return "none"
    except exc as e:
        return "caught " + type(e).__name__
    except Exception as e:
        return "other " + type(e).__name__


for exc in [KeyError, LookupError, ValueError]:
    print(exc.__name__, check(exc, 0), check(exc, 1), check(exc, 2))


class Mine(Exception):
    pass


class Sub(Mine):
    pass


def trap(handler: type[Exception], raised: Exception) -> str:
    try:
        raise raised
    except (handler, TypeError):
        return "trapped"
    except Exception:
        return "missed"


print(trap(Mine, Sub()), trap(Sub, Mine()), trap(Sub, TypeError("t")))
