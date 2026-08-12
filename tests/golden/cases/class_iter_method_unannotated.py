# An unannotated `__iter__` returning `iter(self.items)`. The lazy-iterator
# builtins are synthesized into generator functions by the emitter, so their
# type existed only once that function was emitted -- a body walk that ran
# first read the callee as builtins.object and reported "is not callable",
# which names the compiler's position rather than the program. The walk now
# types them from the same iterationElementType the fold uses.
class Bag:
    def __init__(self) -> None:
        self.items = [1, 2, 3]

    def __iter__(self):
        return iter(self.items)


for x in Bag():
    print(x)

total = 0
for x in Bag():
    total += x
print(total)


# zip, enumerate, reversed and map through the same walk.
class Pair:
    def __init__(self) -> None:
        self.left = [1, 2]
        self.right = ["a", "b"]

    def zipped(self):
        return zip(self.left, self.right)

    def numbered(self):
        return enumerate(self.right)

    def backwards(self):
        return reversed(self.left)

    def doubled(self):
        return map(lambda v: v * 2, self.left)


p = Pair()
for n, s in p.zipped():
    print(n, s)
for i, s in p.numbered():
    print(i, s)
for v in p.backwards():
    print(v)
for v in p.doubled():
    print(v)
