total = 0
for v in (x * 2 for x in range(4)):
    total = total + v
print(total)

words: list[str] = []
for w in (c + "!" for c in "ab" if c != "a"):
    words.append(w)
print(words)

pairs: list[int] = []
for p in (i * 10 + j for i in range(3) for j in range(2)):
    pairs.append(p)
print(pairs)

picked: list[int] = []
for q in (i * j for i in range(3) for j in range(3) if i < j):
    picked.append(q)
print(picked)
print(len(picked))
