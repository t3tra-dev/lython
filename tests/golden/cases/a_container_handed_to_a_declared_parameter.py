# An empty container literal takes its element type from the operations that
# fill it, and handing it to a function that DECLARES what it takes says the
# element as plainly as an append does. Nothing looked:
#
#     h = []
#     put(h, 5)              # def put(heap: "list[int]", value: int)
#     print(h[0] + 1)
#     # builtins.object does not provide manifest method '__add__'
#
# Why execution: the element type decides what may be read back out, so the
# program has to DECODE what the callee put there.
#
# ⭐ Only a parameter that is a CONTAINER of this literal's kind with an
# element of its own. `print(xs)` takes `object` and `len(xs)` takes a
# structural bound; neither says anything about an element, and reading them as
# if they did is the mistake the two protocol repairs beside this one describe.
#
# ⛔ A callee whose parameter is itself INFERRED says nothing here, because
# what it would say is the answer being asked for: `def put(heap, value)` fed
# only by this container is a circle, and it is still refused.


def fill(bucket: "list[str]", word: str) -> None:
    bucket.append(word)


def tally(counts: "dict[str, int]", key: str) -> None:
    if key not in counts:
        counts[key] = 0
    counts[key] = counts[key] + 1


def mark(seen: "set[int]", value: int) -> None:
    seen.add(value)


words = []
fill(words, "ant")
fill(words, "bee")
print(words)
print(words[1] + "!")

counts = {}
tally(counts, "a")
tally(counts, "a")
tally(counts, "b")
print(sorted(counts.items()))
print(counts["a"] + 1)

seen = set()
mark(seen, 3)
mark(seen, 4)
print(sorted(seen))
print(len(seen) + 1)
