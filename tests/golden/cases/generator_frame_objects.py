# Owned values crossing a yield are absorbed by the generator frame (R3):
# the frame words carry each lane's physical span (pointer/size words plus
# the evidence pair for int), transferred in at suspend and back out on
# resume through the frame store/load ownership contracts.
def carries_big():
    total = 10 ** 25
    yield 1
    yield total


for v in carries_big():
    print(v)


def strframe():
    prefix = "acc:"
    yield "one"
    yield prefix + "two"


for s in strframe():
    print(s)


def listframe():
    items: list[int] = []
    items.append(7)
    yield 1
    items.append(35)
    yield items[0] * items[1]


for v in listframe():
    print(v)
