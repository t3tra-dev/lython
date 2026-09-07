# Why execution: every program here was refused or mis-executed, and each of
# the refusals was the same storage decision -- a union field kept its tag and
# every member's lanes in the INSTANCE's own value group. So an object with one
# expanded past the single address a container slot holds, a store rebound an
# SSA value that a later block could not see, and the field's reference was an
# alias of whatever value the last store spliced in.
#
# The field is one payload handle now, and the handle's class word says which
# member it holds -- the reading a `list[int | str]` element has always had.
class Box:
    def __init__(self, v: "int | str") -> None:
        self.v: "int | str" = v


class Holder:
    def __init__(self, b: Box) -> None:
        self.b: Box = b


class Leaf:
    def __init__(self, n: int) -> None:
        self.n: int = n


class Node:
    def __init__(self, v: int) -> None:
        self.v: int = v
        self.nxt: "Node | Leaf" = Leaf(0)


def decode(v: "int | str") -> str:
    if isinstance(v, str):
        return "s:" + v
    return "i:" + str(v + 1)


# A CONTAINER of such objects: the refusal was "a Box value expands to 5
# physical values and nothing can rebuild them from one address".
def containers() -> None:
    boxes = [Box(1), Box("a")]
    print(len(boxes), decode(boxes[0].v), decode(boxes[1].v))
    table = {"k": Box("q")}
    print(decode(table["k"].v))
    pair = (Box(2), Box("z"))
    print(decode(pair[0].v), decode(pair[1].v))
    grown: "list[Box]" = []
    grown.append(Box(5))
    print(len(grown), decode(grown[0].v))
    nested = Holder(Box("in"))
    print(decode(nested.b.v))
    kept = [b for b in boxes if isinstance(b.v, str)]
    print(len(kept))


# A STORE INSIDE A REGION, read after the merge: this produced invalid IR --
# "operand #0 does not dominate this use" -- because the store rebound a value
# whose block does not dominate the read.
def stores_in_regions() -> None:
    one = Box("a")
    if len("x") == 1:
        one.v = 1
    print(decode(one.v))

    two = Box("a")
    for i in range(2):
        two.v = i
    print(decode(two.v))

    three = Box("a")
    n = 0
    while n < 2:
        three.v = n
        n += 1
    print(decode(three.v))

    four = Box("a")
    if len("x") == 1:
        four.v = 1
    else:
        four.v = 2
    print(decode(four.v))

    # The member FLIPS between trips, and the read at the loop head has to see
    # what the last trip stored: this printed the first trip's answer twice.
    five = Box("ab")
    for i in range(2):
        if isinstance(five.v, str):
            print("s", len(five.v))
            five.v = 7
        else:
            print("i", five.v)


# A SECOND STORE, and one that takes a parameter: the field's reference used to
# be the frame's own value, so releasing what the field held released something
# the frame did not own -- refused, or leaked.
def stored_twice(n: int, text: str) -> str:
    b = Box(1)
    b.v = n
    b.v = n + 1
    got = b.v
    b.v = text
    b.v = text
    return decode(got) + "/" + decode(b.v)


# A RECURSIVE union field. `nxt: "Node | Leaf"` used to have no finite layout:
# "class layout contains itself through a union-typed field".
def chain(depth: int) -> Node:
    head = Node(depth)
    if depth > 0:
        head.nxt = chain(depth - 1)
    return head


def total(n: Node) -> int:
    got = n.nxt
    if isinstance(got, Node):
        return n.v + total(got)
    return n.v + got.n


def main() -> None:
    containers()
    stores_in_regions()
    print(stored_twice(4, "hi"))
    print(total(chain(3)))
    # Rendered straight out of the field, with no local and no guard: the read
    # aliased the instance's own lanes, and the frame would not release the
    # instance it had already partly released.
    print(Box(7).v, Box("t").v)


main()
