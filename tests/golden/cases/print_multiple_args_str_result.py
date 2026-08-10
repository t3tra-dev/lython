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


main()
