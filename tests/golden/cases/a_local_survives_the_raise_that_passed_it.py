# What: the raise's dying-local cleanup runs before the raise, so anything it
# frees is gone by the time the handler runs -- and these locals are read
# AFTER the handler. Running it is the only way to see they survived: a freed
# str still has bytes at its address, so the wrong answer here is a plausible
# one rather than a crash.
def report(key: str) -> str:
    problem = KeyError(key)
    prefix = "seen:"
    try:
        raise problem
    except KeyError:
        pass
    return prefix + str(problem)


def collected(key: str) -> str:
    problem = KeyError(key)
    try:
        raise problem
    except KeyError as caught:
        note = str(caught)
    return note + "|" + str(problem)


print(report("a"), report("b"))
print(collected("c"))
