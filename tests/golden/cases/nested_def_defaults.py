# Nested def non-constant defaults: CPython evaluates them when the def
# statement executes — once per ENCLOSING execution, shared by every call of
# that instance, re-evaluated when the enclosing function runs again.
count: int = 0


def stamp() -> int:
    global count
    count += 1
    return count


def outer() -> None:
    def show(tag: int = stamp()) -> None:
        print(tag)

    show()
    show()
    show(99)
    show()


def scaled_defaults(base: int) -> None:
    factor: int = base * 10

    def scaled(x: int = factor + 1) -> int:
        return x

    print(scaled())
    print(scaled(5))


outer()
outer()
scaled_defaults(1)
scaled_defaults(2)
