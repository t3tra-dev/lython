# Owned values crossing a yield are absorbed by the generator frame
# (rfc/stdlib-semantics.md R3): the list's token transfers into the frame at
# suspend, back into the body on resume, and the body keeps mutating it
# across suspensions (runtime-mode append after evidence is severed).
def keeps_list_across_yield():
    items: list[int] = []
    items.append(1)
    yield 1
    items.append(2)
    yield items[0] + items[1]

for v in keeps_list_across_yield():
    print(v)
