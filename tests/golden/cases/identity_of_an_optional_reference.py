# WHAT: `x is y` where x is `Node | None` and y a Node (or a subclass): True
# only when x holds that very object, False when it holds None or another.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: which object the field
# holds, and whether it holds one at all, is a run-time fact.
#
# ⛔ It was refused as "no stable identity", the answer for int and str,
# and `node.parent is root` is how a tree is walked.


class Node:
    def __init__(self, name: str) -> None:
        self.name = name
        self.parent: Node | None = None


class Leaf(Node):
    pass


root = Node("root")
a = Node("a")
a.parent = root
b = Leaf("b")
b.parent = a
print(a.parent is root, a.parent is a, root.parent is root)
print(root is a.parent, a.parent is not root, root.parent is not a)
print(b.parent is a, a.parent is b)
