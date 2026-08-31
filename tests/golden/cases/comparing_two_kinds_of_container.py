# What: `==` between two different kinds of value is False by CPython's
# NotImplemented rule, and the answer is a constant either way -- so running it
# is what shows the fold picked the right constant, and that a WITHIN-kind
# comparison still went the ordinary way and compared contents.
values = [1, 2]
pair = (1, 2)
mapping = {"a": 1}
members = {1, 2}

print(values == 1, mapping == 1, pair == 1, members == 1)
print(values == "ab", values == pair, mapping == values, range(2) == 2)
print(None == values, None == mapping)

print(values == [1, 2], values != [1], pair == (1, 2), mapping == {"a": 1})
print(members == {1, 2}, members != {1}, values == values)


def describe(value: "list[int] | int") -> str:
    if value == 0:
        return "zero"
    return "other"


print(describe(0), describe([1]))
