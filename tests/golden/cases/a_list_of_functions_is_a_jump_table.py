# What: same-signature functions in one container are ONE callable type, so
# calling an element has to reach the element's own body -- which only running
# them shows, since every spelling here answers with a different value.
handlers = [lambda: 1, lambda: 2, lambda: 3]
print(handlers[0](), handlers[1](), handlers[2]())
print([h() for h in handlers])


def double(n: int) -> int:
    return n * 2


def negate(n: int) -> int:
    return -n


ops = [double, negate]
print([op(5) for op in ops])


def pick(flag: bool):
    return (lambda: "yes") if flag else (lambda: "no")


print(pick(True)(), pick(False)())

table = {"a": lambda: "A", "b": lambda: "B"}
print(table["a"](), table["b"]())
