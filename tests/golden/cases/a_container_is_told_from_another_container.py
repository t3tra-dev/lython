# What: isinstance narrows a union of two container classes on both arms --
# the true arm indexes the tuple, the false arm iterates the list. Running it
# is what shows the else arm kept the member type rather than the union.
def total(value: "list[int] | tuple[int, int]") -> int:
    if isinstance(value, tuple):
        return value[0] + value[1]
    out = 0
    for item in value:
        out += item
    return out


def kind(value: "dict[str, int] | list[int]") -> str:
    if isinstance(value, dict):
        return "dict of " + str(len(value))
    return "list of " + str(len(value))


print(total([7, 8]), total((9, 10)))
print(kind({"a": 1}), kind([1, 2, 3]))
