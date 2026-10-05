# WHAT: bytes(list) takes each element as one byte and refuses any outside
# range(0, 256) with CPython's ValueError -- the value 300 printed as b','
# (its low byte) before, and -1 as b'\xff'.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: the refusal depends on the
# runtime values in the list, and the answer is the exception's message.
def build(values: list[int]) -> bytes:
    return bytes(values)


print(build([0, 65, 255]))
for bad in [[300], [-1], [1, 2, 256], [2 ** 70]]:
    try:
        print(build(bad))
    except ValueError as e:
        print("ValueError:", e)
