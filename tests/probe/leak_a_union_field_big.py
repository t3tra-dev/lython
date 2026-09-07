# WHAT: a class field whose type is a union of two OWNING members, stored and
# replaced each trip. The store retains the new arm and releases whatever the
# field held; while the field kept its members INLINE those two references were
# one SSA value, the planner read the field's release as discharging the
# frame's own token too, and the previous str was never freed -- 1 alloc / 58 B
# for a ten-character str, measured by leak_sweep. Sized past the RSS probe's
# floor with 4096-character arms, so one missed release is 4 KB.
class Box:
    def __init__(self, v: "int | str") -> None:
        self.v: "int | str" = v


def once(text: str, n: int) -> int:
    b = Box(text)
    b.v = n
    b.v = text
    held = b.v
    if isinstance(held, str):
        return len(held)
    return 0


i = 0
total = 0
while i < 3000:
    total += once("z" * 4096, i)
    i += 1
print(total > 0)
