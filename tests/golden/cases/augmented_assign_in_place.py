# Why execution: the mutation was invisible, not refused. `b += [3]` built a
# fresh list and rebound `b`, so every other name for that list -- an alias, or
# the caller when the target was a parameter -- still saw the old contents.
# The compiler exited 0 and printing `b` alone looked right, so only reading
# through the OTHER name tells the two apart.
#
# CPython's list.__iadd__ is extend and dict.__ior__ is update; both must
# mutate in place. str and int have no in-place dunder and must keep rebinding.


def through_an_alias() -> None:
    first: list[int] = [1, 2]
    second: list[int] = first
    second += [3]
    print(first)
    print(second)


def through_a_parameter(values: list[int]) -> None:
    values += [3]


def dict_in_place() -> None:
    left: dict[str, int] = {"a": 1}
    alias: dict[str, int] = left
    left |= {"b": 2}
    print(len(alias))


def rebinding_kinds() -> None:
    text: str = "a"
    text += "b"
    print(text)
    count: int = 1
    count += 2
    print(count)


def main() -> None:
    through_an_alias()
    caller: list[int] = [1, 2]
    through_a_parameter(caller)
    print(caller)
    dict_in_place()
    rebinding_kinds()


main()
