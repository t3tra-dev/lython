# What: the yielded list is replaced on the NEXT line, so the loop's carried
# name and the merge's "value that was replaced" are the same object on the
# trip that did not yield. Running it is the only way to see that one of the
# two took the reference and the other did not: the second chunk is where a
# freed list shows up.
def chunks(values: "list[int]", size: int):
    current: list[int] = []
    for v in values:
        current.append(v)
        if len(current) == size:
            yield current
            current = []


for chunk in chunks([1, 2, 3, 4, 5, 6], 2):
    print(chunk)
print(list(chunks([1, 2, 3, 4], 2)))
