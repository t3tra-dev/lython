# What: `if not line: continue` before a yield is the shape that leaves a bare
# `i1` live across the suspension. Running it is what shows the resumption
# picked up where it left off -- the skip has to hold across the yield, and a
# generator that loses it yields the blank lines too.
def nonblank(text: str):
    for line in text.splitlines():
        if not line:
            continue
        yield line


def widths(words: "list[str]"):
    for w in words:
        if not w:
            continue
        yield len(w)


print(list(nonblank("a\n\nb\n\n\nc")))
print(list(widths(["ab", "", "cde", ""])))
for value in nonblank("x\n\ny"):
    print(value)
