# Why execution: same shape as the *args tuple -- the dict a **kwargs
# parameter receives was created with a count and no entries. It failed
# earlier than the tuple did, at the call ("cannot adapt builtins.int to
# runtime input 1"), because the caller had no boxed value to hand the hidden
# lane either. These pin the entries, and the leak gate pins that the dict
# releases what it now retains.


def total(**kwargs: int) -> int:
    acc = 0
    for v in kwargs.values():
        acc += v
    return acc


def by_key(**kwargs: int) -> None:
    for k in sorted(kwargs.keys()):
        print(k, kwargs[k])


def count(**kwargs: int) -> int:
    return len(kwargs)


def whole(**kwargs: int) -> None:
    print(kwargs)


def with_fixed(x: int, **kwargs: str) -> None:
    for k in sorted(kwargs.keys()):
        print(x, k, kwargs[k])


def empty(**kwargs: int) -> int:
    return len(kwargs)


def main() -> None:
    print(total(a=1, b=2))
    by_key(b=2, a=1)
    print(count(a=1, b=2, c=3))
    whole(a=1)
    with_fixed(1, a="p", b="q")
    print(empty())


main()
