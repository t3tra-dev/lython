# WHAT: `text.splitlines(keepends=True)` bound inside a loop and read after it,
# and the same for `split(sep=...)` and `replace(..., count=...)`.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: the read after the loop
# is of the value the last trip bound, which is the keyword call's result.
#
# ⛔ The slot the loop takes for the name is typed by the inference, and the
# inference read the keyword as an argument the manifest's signature has no
# name for: the emitter wrote the call with the keyword moved to its position
# and the slot was never typed, so the read was "unresolved name".


def lines_of(text: str) -> list[str]:
    for _ in range(1):
        lines = text.splitlines(keepends=True)
    return lines


def fields_of(text: str) -> list[str]:
    for _ in range(1):
        fields = text.split(sep=",", maxsplit=1)
    return fields


def replaced(text: str) -> str:
    for _ in range(1):
        out = text.replace("a", "b", count=1)
    return out


print(lines_of("one\ntwo\n"))
print(fields_of("a,b,c"))
print(replaced("aaa"))
