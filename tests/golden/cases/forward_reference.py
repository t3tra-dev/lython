def caller() -> int:
    return callee()


def callee() -> int:
    return 1


def is_even(n: int) -> bool:
    if n == 0:
        return True
    return is_odd(n - 1)


def is_odd(n: int) -> bool:
    if n == 0:
        return False
    return is_even(n - 1)


print(caller())
print(is_even(10))
print(is_odd(7))
