# WHAT: `sep.join(s)` where the operand is a STR. The emitter materializes it
# into a list of one-character strs before the manifest call, so each trip
# allocates that list AND every character in it -- the shape that has to be
# released is the temporary, not the result. Sized past the probe floor with a
# 4096-character operand, which is 4096 one-char strs plus the list per trip.
def run(text: str) -> int:
    return len("-".join(text))


i = 0
total = 0
while i < 300:
    total += run("z" * 4096)
    i += 1
print(total > 0)
