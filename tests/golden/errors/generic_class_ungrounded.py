class Stack[T]:
    def __init__(self) -> None:
        self.items: dict[int, T] = {}


# Nothing determines T: the constructor takes no argument that mentions it,
# and the binding carries no annotation.
s = Stack()
