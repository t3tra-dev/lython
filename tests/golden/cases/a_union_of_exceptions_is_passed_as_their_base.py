# What: a value typed `ValueError | KeyError | Mine` is passed, appended,
# raised and bound where an `Exception` is declared, and is the exception it
# holds: its class, its text, its args and its repr are the active member's.
# Why run: which member is active is a run-time fact.
class Mine(Exception):
    def __str__(self) -> str:
        return "mine!"


def pick(n: int) -> ValueError | KeyError | Mine:
    if n == 0:
        return ValueError("v")
    if n == 1:
        return KeyError("k")
    return Mine("m")


def describe(e: Exception) -> str:
    return type(e).__name__ + ": " + str(e)


def boom(e: BaseException) -> None:
    raise e


es: list[Exception] = []
for i in range(3):
    es.append(pick(i))
    print(describe(pick(i)))
print([str(e) for e in es])
for i in range(3):
    try:
        boom(pick(i))
    except Exception as caught:
        print("caught", type(caught).__name__, caught.args)
held: Exception = pick(2)
print(repr(held))
