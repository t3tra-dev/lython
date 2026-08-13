# Why execution: these printed [1, 3, None] or aborted in repr ("boxed element
# has no conforming __repr__"), and only running them shows either.
#
# A container put INTO another one has two ways to be mutated. The mutation
# through the holder's element takes the runtime arm and stores at the LOADED
# length; the mutation through the original name took the evidence arm, which
# stores at the compile-time element count while taking the new length from
# the runtime word. The two disagreed about where the end of the list is, so
# one overwrote the other and left a hole.


class Box:
    def __init__(self, xs: list[int]) -> None:
        self.xs = xs


def through_a_list_literal() -> None:
    a: list[int] = [1]
    holder: list[list[int]] = [a]
    holder[0].append(2)
    a.append(3)
    print(a, holder[0])


def the_same_list_twice() -> None:
    a: list[int] = [1]
    holder: list[list[int]] = [a, a]
    holder[0].append(2)
    a.append(3)
    print(a, holder[0], holder[1])


def through_a_field() -> None:
    a: list[int] = [1]
    b = Box(a)
    b.xs.append(2)
    a.append(3)
    print(a, b.xs)


# This one kept the file out of the leak gate until 2026-08-14: replacing a
# list element leaked the old one, 3 allocations / 8316 B per execution. The
# element had two references and two releases, and the ownership walk counted
# one reference against them and retained to make up the difference
# (tests/probe/wb_aggregate_slot_unfold_retain_leak.py). Now in the gate.
def through_a_subscript_store() -> None:
    a: list[int] = [1]
    holder: list[list[int]] = [[9]]
    holder[0] = a
    holder[0].append(2)
    a.append(3)
    print(a, holder[0])


def through_a_dict_value() -> None:
    a: list[int] = [1]
    d: dict[str, list[int]] = {"k": a}
    d["k"].append(2)
    a.append(3)
    print(a, d["k"])


def not_shared_at_all() -> None:
    # The evidence arm is still the right one here, and this pins that the
    # mark is not handed out to every list.
    a: list[int] = [1]
    a.append(2)
    a.append(3)
    print(a, a[0], a[2], len(a))


def main() -> None:
    through_a_list_literal()
    the_same_list_twice()
    through_a_field()
    through_a_subscript_store()
    through_a_dict_value()
    not_shared_at_all()


main()
