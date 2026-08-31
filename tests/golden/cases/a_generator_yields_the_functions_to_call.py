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


# A closure the generator builds over its OWN loop target: the cell is a frame
# lane, and the value each function answers with is the only thing that shows
# the lane carried the cell rather than a copy of it.
def closures(n: int):
    for i in range(n):
        yield lambda: i


print([f() for f in list(closures(3))])
print([f() for f in closures(3)])


def named(n: int):
    for i in range(n):
        k = i * 10
        yield lambda: k


print([f() for f in list(named(3))])


# A def declared in the loop body is a name the yield walk has to know the
# type of, and the closure it makes reads the frame's cell like any other.
def declared(n: int):
    for i in range(n):
        def step() -> int:
            return i + 100
        yield step


print([f() for f in list(declared(3))])
