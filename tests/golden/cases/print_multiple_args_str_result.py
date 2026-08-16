# Why execution: the defect was quoting. `print(max("a", "b"), 1)` printed
# 'b' 1 where CPython prints b 1 -- a str rendered through repr, which quotes.
# The compiler exited 0 and a compile check would not see it.
#
# `max` over two arguments is folded here into a comparison and a select, so
# the emitted value is a str; the general inference does not model that fold
# and answered `builtins.object`, which routed an already-str down the repr
# path. Single-argument print renders differently, which is why only the
# multi-argument form was wrong.


def main() -> None:
    print(max("a", "b"), 1)
    print(min("a", "b"), 1)
    words: list[str] = ["pear", "fig"]
    print(min(words), max(words))
    print(max(1, 2), 3)
    print("plain", "strings")
    print(max("a", "b"))

    # ⭐ `sep=` is this ladder's own separator. Any keyword at all used to make
    # the whole ladder decline, and the call then landed on `builtins.print`'s
    # contract -- which has no keyword parameters, so the report was "call
    # arguments do not match the Callable contract" with `sep` named nowhere.
    # The join here already builds the space-separated string CPython's default
    # produces; a different separator is a different constant.
    print("a", "b", sep="-")
    print(1, 2, 3, sep="")
    print(1, 2, sep=" | ")
    print("only", sep="-")
    dash = "=="
    print("x", "y", sep=dash)
    print(max("a", "b"), 1, sep="/")


main()
