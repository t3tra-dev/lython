# WHAT: a name bound inside a loop and read after it. The binding lives in a
# slot the scope allocates before the loop, so the slot and the string it holds
# both have to be released once per call; the string is sized past the probe
# floor.
def pick(n: int) -> str:
    for v in [n]:
        chosen = "z" * 4096 + str(v)
    return chosen


i = 0
total = 0
while i < 4000:
    total += len(pick(i))
    i += 1
print(total)
