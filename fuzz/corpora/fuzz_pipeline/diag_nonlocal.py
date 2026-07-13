def outer() -> int:
    n = 0

    def inner() -> int:
        nonlocal n
        n += 1
        return n

    return inner()


print(outer())
