# WHAT: a starred call whose tuple comes through a binding, so the expansion is
# a runtime read of the payload rather than the literal's own evidence. Each
# member is borrowed from the tuple; the string is sized past the probe floor.
def take(a: str, b: int) -> int:
    return len(a) + b


t: "tuple[str, int]" = ("z" * 4096, 3)
i = 0
total = 0
while i < 4000:
    total += take(*t)
    i += 1
print(total)
