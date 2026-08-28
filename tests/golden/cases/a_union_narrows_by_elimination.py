# WHAT: a three-member union narrowed by ELIMINATION -- each guard rules one
# member out and returns, so the tail is the one that is left.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: which member the tail is
# reading is the value. The `is None` guard leaves a two-member union that no
# single unwrap can express, so what the next guard proves has to survive as a
# TYPE; getting that wrong produces a program that reads the union's other
# member and prints its rendering.
import sys


def render(v: "int | str | None") -> str:
    if v is None:
        return "none"
    if isinstance(v, int):
        return "int:" + str(v + 1)
    # Two guards have run: `v` is a str, and the concatenation says so.
    return "str:" + v.upper() + "/" + str(len(v))


sys.stdout.write(render(7) + "\n")
sys.stdout.write(render("ab") + "\n")
sys.stdout.write(render(None) + "\n")


# The other order, and a member that is itself a container.
def describe(v: "float | str | None") -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return "f" + str(v * 2)
    return "s" + v


sys.stdout.write(describe(1.5) + " " + describe("x") + " " + describe(None)
                 + "\n")


# Four members: two eliminations still leave a union, and the third is what
# the tail reads.
def four(v: "int | float | str | None") -> str:
    if v is None:
        return "n"
    if isinstance(v, int):
        return "i"
    if isinstance(v, float):
        return "f"
    return "s" + v


sys.stdout.write(four(1) + four(1.5) + four("q") + four(None) + "\n")
