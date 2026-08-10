# Why execution: the length was already right and only the element read was
# wrong -- `len(data[0])` returned 2 while `data[0][1]` raised IndexError. The
# outer list's cached description of the inner one still said "one element",
# and the subscript was answered from that rather than from the payload the
# callee had grown. Nothing but running the program sees the difference.


def grow(values: list[int]) -> None:
    values.append(2)


def grow_dict_value(values: list[int]) -> None:
    values.append(9)


def main() -> None:
    data: list[list[int]] = [[1]]
    grow(data[0])
    print(len(data[0]))
    print(data[0][1])
    print(data)

    through_a_local: list[list[int]] = [[1]]
    inner = through_a_local[0]
    grow(inner)
    print(inner[1])
    print(through_a_local[0][1])

    plain: list[int] = [1]
    grow(plain)
    print(plain[1])

    mapping: dict[str, list[int]] = {"k": [1]}
    grow_dict_value(mapping["k"])
    print(mapping["k"][1])


main()
