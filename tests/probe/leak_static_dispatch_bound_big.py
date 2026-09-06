# WHAT: a `@staticmethod` an override sits behind, READ off a base-typed
# receiver in a loop. The read builds a forwarder that captures the receiver,
# so the closure store and the instance it holds both have to be released; the
# instance carries a string sized past the probe floor.
class Shape:
    tag: str

    def __init__(self, tag: str) -> None:
        self.tag = tag

    @staticmethod
    def scale(n: int) -> int:
        return n


class Square(Shape):
    @staticmethod
    def scale(n: int) -> int:
        return n * 4


i = 0
while i < 3000:
    s: Shape = Square("z" * 4096)
    m = s.scale
    m(3)
    i += 1
print("done")
