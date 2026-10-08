# WHAT: id() answers what `is` answers: the same object has one id wherever it
# is read from -- a name, a list slot, an `object` parameter -- and two objects
# have two; the id of an instance is the address its default repr prints; None
# and True have one id each; the argument is evaluated once.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: identity is an address the
# program reads at run time; a box per read would compile and answer False.

class A:
    pass

a = A()
b = a
print(id(a) == id(b), id(a) == id(A()))
xs = [a, A()]
print(id(xs[0]) == id(a), id(xs[1]) == id(a))
print(hex(id(a)) in repr(a))
def same(p: object, q: object) -> bool:
    return id(p) == id(q)
print(same(a, a), same(a, xs[1]), same(None, None), same(True, True))
d: dict[str, int] = {}
e = d
print(id(d) == id(e), id(d) == id({}), id(None) == id(None))
seen: set[int] = set()
for o in [a, b, xs[1]]:
    seen.add(id(o))
print(len(seen))
x: object = None
print(id(x) == id(None))
def ret() -> None:
    print("called")
print(id(ret()) == id(None))
