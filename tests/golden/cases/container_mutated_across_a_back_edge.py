# Why execution: every one of these compiled and printed a WRONG VALUE, and two
# spellings aborted at runtime instead. The bug is that a read was answered
# from the container's definition-time contents, so nothing but running it
# shows the accumulated value. In the leak gate too: the computed-index form
# used to retain and release different objects, which drove a live string to
# refcount zero.


def list_constant_index() -> int:
    xs: list[int] = [0, 0]
    for _ in range(3):
        xs[0] += 1
    return xs[0]


def list_computed_index() -> list[int]:
    xs: list[int] = [0, 0]
    for i in range(4):
        xs[i % 2] += 1
    return xs


def dict_value() -> str:
    d: dict[str, str] = {"a": ""}
    for _ in range(3):
        d["a"] += "x"
    return d["a"]


def strings_read_each_time() -> None:
    xs: list[str] = ["", ""]
    for _ in range(3):
        xs[0] += "x"
        print(xs[0])


def while_loop() -> int:
    xs: list[int] = [0, 0]
    i = 0
    while i < 3:
        xs[0] += 1
        i += 1
    return xs[0]


def append_then_read() -> int:
    xs: list[int] = [1]
    for i in range(3):
        xs.append(i)
    return len(xs)


def main() -> None:
    print(list_constant_index())
    print(list_computed_index())
    print(dict_value())
    strings_read_each_time()
    print(while_loop())
    print(append_then_read())


main()
