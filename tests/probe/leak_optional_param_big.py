# WHAT: rewriting an optional field through a parameter has to release the
# payload it replaced. The store lands one frame down from the object's owner,
# which is the path that had no release at all before boxing reached it.
class Box:
    s: "str | None"

    def __init__(self) -> None:
        self.s = None


def put(b: Box, v: "str | None") -> None:
    b.s = v


b = Box()
i = 0
while i < 4000:
    put(b, "y" * 4096)
    put(b, None)
    put(b, "z" * 4096)
    i += 1
print("done")
