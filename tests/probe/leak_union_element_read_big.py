# WHAT: reading a union element out of a container. Each read rebuilds the
# member's lanes from the container's own box and borrows them, so nothing
# should be retained per read; the string member is sized past the probe floor.
xs: "list[int | str]" = ["z" * 4096, 7]
d: "dict[str, int | str]" = {"k": "y" * 4096}
i = 0
total = 0
while i < 3000:
    for v in xs:
        if isinstance(v, str):
            total += len(v)
        else:
            total += v
    key = "k"
    w = d[key]
    if isinstance(w, str):
        total += len(w)
    i += 1
print(total)
