# Refcount stress for the pop/insert paths, with heap-allocated elements so a
# missing release leaks and an extra one aborts at teardown (exit 134). The
# shapes are the ones that move the most boxes: pop(0)/insert(0) at the front
# (maximum shifting), a full drain, nested lists whose inner list must be
# released exactly once, and repeated middle inserts (growth and shifting
# together).
words: list[str] = []
i: int = 0
while i < 40:
    words.append("item" + str(i))
    i = i + 1

j: int = 0
while j < 20:
    head: str = words.pop(0)
    words.insert(0, head + "!")
    j = j + 1
print(len(words), words[0], words[19])

acc: int = 0
while len(words) > 0:
    tail: str = words.pop()
    acc = acc + len(tail)
print(acc, len(words))

rows: list[list[int]] = []
k: int = 0
while k < 15:
    rows.append([k, k + 1, k + 2])
    k = k + 1
total: int = 0
while len(rows) > 0:
    row: list[int] = rows.pop(0)
    total = total + len(row)
print(total, len(rows))

nums: list[int] = [0]
m: int = 0
while m < 30:
    nums.insert(1, m)
    m = m + 1
print(len(nums), nums[0], nums[1], nums[30])
