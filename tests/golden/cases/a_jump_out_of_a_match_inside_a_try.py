# The guard that decides whether a loop body's `try` carries a break or a
# continue walked `body`/`orelse`/`finalbody`/`handlers` by name -- and a
# `match` keeps its statements under `cases`, so a jump inside a match arm was
# invisible to it. The loop was then built as if it had none, and the program
# died in the lowering with "reference to block defined in another region".
# Must run: what the guard decides is which loop shape gets built, and only the
# printed order says the finally still ran on the way out.


def until_two(values: list[int]) -> str:
    out = ""
    for v in values:
        try:
            match v:
                case 1:
                    out = out + "one"
                case 2:
                    break
                case _:
                    out = out + "?"
        finally:
            out = out + "."
    return out


print(until_two([1, 3, 2, 1]))


def skip_two(values: list[int]) -> str:
    out = ""
    for v in values:
        try:
            match v:
                case 2:
                    continue
                case _:
                    out = out + str(v)
        finally:
            out = out + "."
    return out


print(skip_two([1, 2, 3]))


# The jump one level further in: a match arm inside an `if` inside the try.
def nested(values: list[int]) -> str:
    out = ""
    for v in values:
        try:
            if v > 0:
                match v:
                    case 3:
                        break
                    case _:
                        out = out + str(v)
        finally:
            out = out + "|"
    return out


print(nested([1, 2, 3, 4]))
