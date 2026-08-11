# Why execution: the tuple a *args parameter receives was CREATED EMPTY --
# `__new__` is told the arity and nothing else -- and the values reached the
# callee only through hidden per-element lanes that exist for two consumers.
# Everything else read uninitialized memory, so these printed nothing, or
# garbage digits, or aborted in Ly_IncRef. `len(args)` was right throughout,
# which is exactly why only running the rest of them shows it. In the leak
# gate too: the tuple now retains what it holds.


def count(*args: int) -> int:
    return len(args)


def each(*args: int) -> None:
    for a in args:
        print(a)


def total(*args: int) -> int:
    acc = 0
    for a in args:
        acc += a
    return acc


def whole(*args: int) -> None:
    print(args)


def with_fixed(a: int, *rest: int) -> int:
    return a + len(rest)


def strings(*args: str) -> None:
    for s in args:
        print(s.upper())


def by_index(*args: int) -> None:
    print(args[0], args[1])


def empty(*args: int) -> int:
    return len(args)


def main() -> None:
    print(count(1, 2, 3))
    each(1, 2)
    print(total(1, 2, 3))
    whole(1, 2)
    print(with_fixed(1, 2, 3))
    strings("a", "b")
    by_index(7, 8)
    print(empty())


main()
