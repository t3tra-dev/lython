# What: CPython leaves the for target bound after the loop, and the value is
# the last one it took -- so this has to RUN to show the binding reached the
# read at all, and that it carries the final iteration rather than the first.
def last_matching_index(xs: "list[int]", want: int) -> int:
    for i in range(len(xs)):
        if xs[i] == want:
            break
    return i


for value in [10, 20, 30]:
    pass
print(value)

for ch in "abc":
    pass
else:
    print("else saw", ch)

for key, count in [("a", 1), ("b", 2)]:
    pass
print(key, count)

print(last_matching_index([5, 6, 7], 6))
