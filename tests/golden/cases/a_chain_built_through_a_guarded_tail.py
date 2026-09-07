# Why execution: the answer is a REFERENCE COUNT and the failure was a crash --
# "Ly_DecRef observed non-positive refcount" from the second trip on, with no
# diagnostic before it. Every function here builds a chain the way Python
# builds one, then walks it, so a node freed early shows up as a wrong sum or
# a crash rather than as a compile error.
#
#     tail: "N | None" = None
#     for v in xs:
#         node = N(v)
#         if tail is not None:
#             tail.nxt = node      # the store sits in the GUARDED arm
#         tail = node              # and node is used again below it
#
# A store MOVES the frame's token into the slot when nothing else needs the
# value, and "nothing else needs it" was asked as "does the store DOMINATE a
# later use". A conditional store dominates nothing after the join, so the
# answer was always no, the token moved, and the local was left holding a spent
# one.
#
# ⛔ The same class with a non-union field was correct all along (`tail.v = v`,
# `tail.kids.append(v)`), and so was storing something OTHER than the value the
# loop carries -- which is what made this look like a union-field defect rather
# than a question about where the store sits.
class N:
    def __init__(self, v: int) -> None:
        self.v = v
        self.nxt: "N | None" = None


def build(xs: list[int]) -> "N | None":
    head: "N | None" = None
    tail: "N | None" = None
    for v in xs:
        node = N(v)
        if head is None:
            head = node
            tail = node
        else:
            if tail is not None:
                tail.nxt = node
            tail = node
    return head


def build_while(n: int) -> "N | None":
    head: "N | None" = None
    tail: "N | None" = None
    i = 0
    while i < n:
        node = N(i)
        if tail is not None:
            tail.nxt = node
        else:
            head = node
        tail = node
        i += 1
    return head


def build_through_a_second_name(xs: list[int]) -> "N | None":
    head: "N | None" = None
    tail: "N | None" = None
    for v in xs:
        node = N(v)
        other = node
        if tail is not None:
            tail.nxt = other
        else:
            head = node
        tail = node
    return head


def total(head: "N | None") -> int:
    out = 0
    cur = head
    while cur is not None:
        out += cur.v
        cur = cur.nxt
    return out


def length(head: "N | None") -> int:
    n = 0
    cur = head
    while cur is not None:
        n += 1
        cur = cur.nxt
    return n


def values(head: "N | None") -> list[int]:
    out: list[int] = []
    cur = head
    while cur is not None:
        out.append(cur.v)
        cur = cur.nxt
    return out


def main() -> None:
    chain = build([1, 2, 3, 4])
    print(total(chain), length(chain), values(chain))
    print(total(build([])), length(build([])), values(build([7])))
    print(values(build_while(5)), total(build_while(5)))
    print(values(build_through_a_second_name([9, 8, 7])))
    running = 0
    for size in range(1, 40):
        running += total(build(list(range(size))))
    print(running)


main()
