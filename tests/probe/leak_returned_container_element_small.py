# WHAT: a borrowed value parked in a locally built container and then RETURNED
# by element. The read mints the frame's token and the return transfers it, so
# exactly one retain must stand -- the borrowed-return walk used to add a
# second, which the affine verifier caught, and dropping that one must not swing
# the other way into a release the caller never gets. The operand is a 4096-byte
# str so one leaked reference is well past the probe floor.
def through(text: str) -> str:
    box: list[str] = []
    box.append(text)
    return box[0]


def through_a_tuple(text: str) -> str:
    pair = (text, "z")
    return pair[0]


i = 0
total = 0
while i < 300:
    payload = "q" * 4096 + str(i)
    total += len(through(payload)) + len(through_a_tuple(payload))
    i += 1
print(total > 0)
