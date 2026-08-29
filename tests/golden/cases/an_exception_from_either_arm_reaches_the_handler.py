# A raising call in EACH arm of an `if` inside one `try`, plus the same shape
# one level down (a call whose two arms both raise) and a loop around it. Both
# arms must reach the handler, and the traceback must name the arm that raised.
def parse(text: str, mode: int) -> int:
    try:
        if mode == 1:
            value = int(text)
        else:
            value = int(text) * 2
        return value
    except ValueError:
        return -1


# The reduction that named the defect: both arms call the SAME function on
# DIFFERENT literals, so the two blocks are identical apart from their operands
# and merge -- which is what put the marker id in a phi. `parse` above does not
# reach it (its arms differ by the `* 2`) and neither does an arm pair that is
# identical outright, so this exact shape is the one that has to be here.
def literal(mode: int) -> str:
    try:
        if mode == 1:
            int("zz")
        else:
            int("yy")
        return "no error"
    except ValueError as e:
        return "caught " + str(e)


def classify(text: str) -> str:
    try:
        if len(text) > 2:
            n = int(text)
        else:
            n = int(text)
        return "int " + str(n)
    except ValueError:
        return "not an int: " + text


print(literal(1))
print(literal(2))

for arg in ["7", "zz", "13", "yy"]:
    print(arg, parse(arg, 1), parse(arg, 2), classify(arg))

total = 0
for i in range(4):
    try:
        if i % 2 == 0:
            total += int("10")
        else:
            total += int("bad")
    except ValueError:
        total += 1
print("total", total)

try:
    if len("abc") == 3:
        raise KeyError("from the taken arm")
    else:
        raise KeyError("from the other arm")
except KeyError as e:
    print("KeyError", e)
