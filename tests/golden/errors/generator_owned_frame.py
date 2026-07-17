def keeps_list_across_yield():
    items: list[int] = []
    items.append(1)
    yield 1
    items.append(2)
    yield items[0] + items[1]

for v in keeps_list_across_yield():
    print(v)
