# WHAT: a local created OUTSIDE a loop stored into a field INSIDE it. The store
# runs once per trip against one definition, so it must RETAIN rather than move
# the token -- and a retain with no matching release is exactly what this
# measures. The string is sized past the probe floor and the holder is dropped
# every iteration, so both halves have to balance.
class Holder:
    def __init__(self) -> None:
        self.tag: str = ""


i = 0
while i < 300:
    w: str = "z" * 4096
    h = Holder()
    j = 0
    while j < 3:
        h.tag = w
        j += 1
    if len(h.tag) == 0:
        print("never")
    i += 1
print("done")
