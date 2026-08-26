# probe: leak -- optional class field rebound to a payload and back to None (40000 iterations)
# axes: op=leak-loop iterations=40000
# CPython 3.14 expects: 40000

class Node:
    val: int
    nxt: "Node | None"

    def __init__(self, val: int) -> None:
        self.val = val
        self.nxt = None


def once() -> int:
    head = Node(1)
    head.nxt = Node(2)
    head.nxt = Node(3)
    head.nxt = None
    return head.val


total = 0
for _ in range(40000):
    total += once()
print(total)
