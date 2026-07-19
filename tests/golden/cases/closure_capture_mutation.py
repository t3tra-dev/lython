# In-place container mutation through capture and parameter boundaries: the
# callee sees the runtime payload (not compile-time evidence), and the caller
# observes the callee's stores afterwards.
def set_first(xs: list[int]) -> None:
    xs[0] = 99


def del_last(xs: list[int]) -> None:
    del xs[-1]


def captured() -> None:
    xs: list[int] = [1, 2, 3]

    def poke() -> None:
        xs[0] = 10
        xs[-1] = 30

    poke()
    print(xs[0], xs[1], xs[2])
    print(len(xs))


def parameters() -> None:
    xs: list[int] = [1, 2, 3]
    set_first(xs)
    print(xs[0])
    del_last(xs)
    print(len(xs), xs[-1])


def oob() -> None:
    xs: list[int] = [1]

    def poke() -> None:
        xs[5] = 0

    try:
        poke()
    except IndexError as exc:
        print("caught:", exc)


captured()
parameters()
oob()
