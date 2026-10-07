# WHAT: an unhashable dict key or set element raises CPython 3.14's message
# -- "cannot use 'list' as a dict key (unhashable type: 'list')", "... as a
# set element (...)" -- naming the KEY's class first and the item that refused
# second when a tuple holds it; hash() keeps the bare "unhashable type:
# 'list'"; and tuple keys still hash, find and deduplicate as before.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: the refusal is raised by
# the runtime's dict and set while hashing a boxed key, and its text is built
# from the class-name table at that moment; nothing before execution has it.

def attempt(k: int) -> None:
    try:
        if k == 0:
            print({[1]: 2})
        elif k == 1:
            print({[1], [2]})
        elif k == 2:
            print(frozenset([[1]]))
        elif k == 3:
            print(dict([([1], 2)]))
        elif k == 4:
            print(set([bytearray(b"a")]))
        elif k == 5:
            print({bytearray(b"a"): 1})
        elif k == 6:
            print(set([{1}]))
        elif k == 7:
            print({(1, [2]): 3})
        elif k == 8:
            print({(1, (2, [3]))})
        elif k == 9:
            print(hash((1, [2])))
        elif k == 10:
            print(hash([1]))
        elif k == 11:
            print([1] in {1: 2})
        elif k == 12:
            print([1] in {1, 2})
        elif k == 13:
            print(dict([((1, [2]), 3)]))
        elif k == 14:
            print({(1, 2): 3}[(1, 2)], len({(1, 2), (1, 2), (2, 1)}))
    except TypeError as e:
        print(e)
for k in range(15):
    attempt(k)

class P:
    def __init__(self, a: list[int]) -> None:
        self.a = a
    def __hash__(self) -> int:
        return hash((1, len(self.a)))
    def __eq__(self, o: object) -> bool:
        return isinstance(o, P) and o.a == self.a
class Q:
    def __init__(self, a: int) -> None:
        self.a = a
    def __hash__(self) -> int:
        return hash((self.a, 2))
d = {P([1]): 1, Q(1): 2}
print(len(d), d[P([1])])
s = {Q(3), Q(3)}
print(len(s))
try:
    print({(Q(1), [2]): 1})
except TypeError as e:
    print(e)
try:
    print(hash((Q(1), [2])))
except TypeError as e:
    print(e)
