# What: `:=` binds in the scope AROUND the comprehension (PEP 572), so the name
# is readable after it and holds the value from the LAST element that reached
# the assignment -- which only running it can show. All four spellings are
# here: list, set and dict comprehensions, and a generator expression fused
# into a reducer.
values = [1, 2, 3, 4]

doubled = [y for value in values if (y := value * 2) > 4]
print(doubled, y)

words = ["a", "bb", "ccc"]
print([n for word in words if (n := len(word)) > 1], n)

print(sorted({(k := value * 2) for value in values}), k)
seen = {(key := str(value)): value for value in values}
print(sorted(seen.items()), key)
print(sum((total := value + 1) for value in values), total)


def largest(xs: "list[int]") -> int:
    best = 0
    for x in xs:
        if (scaled := x * 3) > best:
            best = scaled
    return best + scaled


print(largest([1, 5, 2]))
