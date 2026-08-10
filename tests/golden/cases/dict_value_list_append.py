# Why execution: the defect was a silent wrong VALUE. A list reached through a
# dict subscript carries no compile-time element evidence, and the append was
# lowered by the tier that derives the store index from the evidence it has --
# zero -- so it overwrote element 0 while the length grew past a slot nobody
# wrote. Reading that slot gave a garbage integer, a refcount abort or a
# segfault from the same binary. Only running tells the right answer from the
# wrong one; the compiler exited 0 either way.


def append_to_a_non_empty_value() -> None:
    d: dict[str, list[int]] = {"x": [1]}
    d["x"].append(2)
    print(d["x"][0])
    print(d["x"][1])
    print(len(d["x"]))


def append_to_an_empty_value() -> None:
    d: dict[str, list[int]] = {"x": []}
    d["x"].append(5)
    print(d["x"][0])


def group_by() -> None:
    groups: dict[str, list[int]] = {}
    groups["even"] = []
    groups["odd"] = []
    for value in range(6):
        if value % 2 == 0:
            groups["even"].append(value)
        else:
            groups["odd"].append(value)
    print(len(groups["even"]))
    print(groups["even"][2])
    print(groups["odd"][2])


def main() -> None:
    append_to_a_non_empty_value()
    append_to_an_empty_value()
    group_by()


main()
