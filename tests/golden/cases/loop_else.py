for i in range(3):
    print(i)
else:
    print("for-else ran")
for i in range(5):
    if i == 2:
        print("breaking")
        break
else:
    print("not printed")
total = 0
for v in [5, 6]:
    total = total + v
else:
    total = total + 100
print(total)
i = 0
while i < 3:
    i = i + 1
else:
    print("while-else ran", i)
j = 0
found = -1
while j < 10:
    if j == 3:
        found = j
        break
    j = j + 1
else:
    found = -2
print(found)
k = 10
while k < 3:
    k = k + 1
else:
    print("empty while-else", k)
