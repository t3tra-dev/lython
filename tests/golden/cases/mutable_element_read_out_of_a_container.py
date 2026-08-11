# Why execution: reading a mutable container out of another one and appending
# through both records used to write past each other -- an element overwritten
# and a slot never written -- which repr then walked. Five of these aborted
# ("boxed element has no conforming __repr__") and the rest printed
# [1, 2, 4, None]. Only running them shows either.


def list_in_list() -> None:
    t: list[list[int]] = [[1, 2]]
    t[0].append(3)
    a = t[0]
    a.append(4)
    print(a)


def list_in_tuple() -> None:
    t: tuple[list[int], int] = ([1, 2], 5)
    t[0].append(3)
    a, b = t
    a.append(4)
    print(a)


def both_records_agree() -> None:
    t: tuple[list[int], dict[str, int]] = ([1, 2], {"a": 1})
    t[0].append(3)
    t[1]["b"] = 2
    a, b = t
    a.append(4)
    print(t[0], a)
    print(t[1], b)


class Mid:
    leaves: list[int]

    def __init__(self) -> None:
        self.leaves = []


class Top:
    mid: Mid

    def __init__(self) -> None:
        self.mid = Mid()


def two_level_field_chain() -> None:
    t = Top()
    t.mid.leaves.append(1)
    t.mid.leaves.append(2)
    print(t.mid.leaves[1], t.mid.leaves)


def dict_value() -> None:
    d: dict[str, list[int]] = {"a": [1]}
    d["a"].append(2)
    v = d["a"]
    v.append(3)
    print(d["a"], v)


def nested_printed_whole() -> None:
    t: list[list[int]] = [[1, 2], [9]]
    t[0].append(3)
    a = t[0]
    a.append(4)
    print(t)


def main() -> None:
    list_in_list()
    list_in_tuple()
    both_records_agree()
    two_level_field_chain()
    dict_value()
    nested_printed_whole()


main()
