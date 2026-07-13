total = 0
for v in (x * x for x in range(6) if x % 2 == 0):
    total = total + v
print(total)

words: list[str] = []
for w in (c + "!" for c in "abc" if c != "b"):
    words.append(w)
print(words)
