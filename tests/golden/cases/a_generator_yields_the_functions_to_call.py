# What: the function object a generator yields has to be the one the call site
# dispatches to, and only calling it shows which body ran -- each of these
# answers with a different value.
def five() -> int:
    return 5


def six() -> int:
    return 6


def numbers():
    yield five
    yield six


for f in numbers():
    print(f())


def anonymous():
    yield lambda: 1
    yield lambda: 2


print([f() for f in anonymous()])


def texts():
    yield lambda: "a"
    yield lambda: "b"


print([f() for f in texts()])


def doubler(n: int) -> int:
    return n * 2


def with_arguments():
    yield doubler


for f in with_arguments():
    print(f(21))
