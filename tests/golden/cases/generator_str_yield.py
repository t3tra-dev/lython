# str values cross the generator suspension ABI on the object-family value
# lane: the yielded str's physical span transfers to the resumer through the
# clone's owned-results contract (for-loop and explicit next() consumers).
def words(n: int):
    i = 0
    while i < n:
        yield "word" + str(i)
        i = i + 1


def greeting():
    yield "hello"
    yield "world"


for w in words(3):
    print(w)

g = greeting()
print(next(g))
print(next(g))
print("done")
