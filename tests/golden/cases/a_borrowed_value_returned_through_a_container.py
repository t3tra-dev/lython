# Why execution: the answer is a REFERENCE COUNT, and only running can show it
# is neither one too many (a leak) nor one too few (a use after free). The
# program was refused outright -- "borrowed entry argument 0 of @f is returned
# with 2 retained ownership tokens; exactly one may be transferred" -- because
# a container element read MINTS the frame's token by retaining, and the
# borrowed-return walk then added a second retain for the same transfer. The
# value must survive being read back after its container dies, and the caller
# must be the only owner left, so each of these returns is used twice.
#
# ⛔ Every operand here is BORROWED by the function: a parameter, or something
# derived from one. A value the frame produces itself (`xs.append(g())`) never
# reached that walk and always worked, which is why nothing caught this.
def through_a_list(text: str) -> str:
    box: list[str] = []
    box.append(text)
    return box[0]


def through_a_tuple(text: str) -> str:
    pair = (text, "tail")
    return pair[0]


def through_a_literal(value: int) -> int:
    box = [value]
    return box[0]


def past_another_element(value: int) -> int:
    box: list[int] = [0]
    box.append(value)
    return box[1]


def through_two_appends(value: int) -> int:
    box: list[int] = []
    box.append(value)
    box.append(value)
    return box[0]


def through_a_name(text: str) -> str:
    box: list[str] = []
    box.append(text)
    out = box[0]
    return out


class Point:
    def __init__(self, n: int) -> None:
        self.n = n


def an_instance_through_a_list(point: Point) -> Point:
    box: list[Point] = []
    box.append(point)
    return box[0]


def main() -> None:
    word = "borrowed"
    first = through_a_list(word)
    print(first, len(first), first == word)
    second = through_a_tuple(word)
    print(second, second + "!", len(second))
    print(through_a_literal(7), through_a_literal(7) + 1)
    print(past_another_element(9), past_another_element(9) * 2)
    print(through_two_appends(3), through_two_appends(3) - 1)
    named = through_a_name(word)
    print(named, named.upper())
    origin = Point(4)
    same = an_instance_through_a_list(origin)
    print(same.n, same is origin, origin.n)
    total = 0
    for index in range(50):
        total += through_a_literal(index) + len(through_a_list("x" * index))
    print(total)


main()
