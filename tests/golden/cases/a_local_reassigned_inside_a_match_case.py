# What: a local bound BEFORE a `match` and reassigned inside a case body. Each
# case is emitted in a scope of its own -- a value defined in one case's block
# does not dominate the next case's -- so the write landed in a scope that is
# popped when the case ends and the name kept what it held before the match.
#
# WHY THIS IS RUN: which arm executed is a runtime fact, and the value the arm
# wrote is only visible after the match. The decode is that every line prints
# the pre-match value beside the written one: a compiler that drops the write
# prints a plausible answer for the shape and the wrong one for the program,
# and the `if` spelling of the same question beside it is the control.
def classify(n: int) -> str:
    label = "unset"
    match n:
        case 0:
            label = "zero"
        case 1 | 2:
            label = "small"
    return label


def classify_if(n: int) -> str:
    label = "unset"
    if n == 0:
        label = "zero"
    elif n == 1 or n == 2:
        label = "small"
    return label


print(classify(0), classify(2), classify(9))
print(classify_if(0), classify_if(2), classify_if(9))


def running(xs: "list[int]") -> int:
    total = 0
    for x in xs:
        match x:
            case 0:
                total = total
            case n:
                total = total + n
    return total


print(running([1, 0, 2]), running([]))


def guarded(n: int) -> str:
    text = "big"
    match n:
        case v if v < 0:
            text = "neg"
        case 0:
            text = "zero"
    return text


print(guarded(-1), guarded(0), guarded(7))
