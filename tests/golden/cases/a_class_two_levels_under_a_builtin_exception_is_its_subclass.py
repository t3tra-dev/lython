# A class of the program declared under a builtin exception is a subclass of
# every builtin exception above the base it names, not only of that base.
#
# Why it runs: the wrong answer folded `isinstance` to False and compiled the
# arm away, so only the executed result says which arm survived.
class Err(Exception):
    pass


class Deep(ValueError):
    pass


print(issubclass(Err, Exception), issubclass(Err, BaseException))
print(issubclass(Deep, ValueError), issubclass(Deep, Exception),
      issubclass(Deep, BaseException))
print(issubclass(Deep, KeyError))


def classify(e: BaseException) -> str:
    if isinstance(e, Err):
        return "err"
    if isinstance(e, Deep):
        return "deep"
    return "other"


print(classify(Err("x")), classify(Deep("y")), classify(KeyError("z")))


def last_of(n: int) -> BaseException:
    last: BaseException = Err("first")
    for i in range(n):
        last = Deep("again %d" % i)
    return last


print(type(last_of(2)).__name__, last_of(2))
