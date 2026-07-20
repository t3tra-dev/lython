from itertools import count, islice, accumulate

for i in islice(count(), 5):
    print(i)
for i in islice(count(10, 3), 1, 5, 2):
    print(i)
for v in islice(accumulate(count(1)), 2, 5):
    print(v)
